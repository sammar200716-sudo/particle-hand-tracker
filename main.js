import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// --- CONFIGURATION ---
const NUM_PARTICLES = 3000;
const Z_PLANE = 50; // Distance of the camera
const PARTICLE_COLOR = new THREE.Color(0xffffff);
const SCATTER_COLOR = new THREE.Color(0xff3377);
const COLOR_BLUE = new THREE.Color(0x00f0ff);   // left
const COLOR_YELLOW = new THREE.Color(0xffe600); // right
const COLOR_ORANGE = new THREE.Color(0xff7300); // up
const COLOR_PURPLE = new THREE.Color(0xb000ff); // down

// --- DOM ELEMENTS ---
const video = document.getElementById('webcam');
const canvasContainer = document.getElementById('canvas-container');
const statusText = document.getElementById('status-text');
const handStateText = document.getElementById('hand-state');
const statusIndicator = document.querySelector('.status-indicator');

// --- THREE.JS SETUP ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = Z_PLANE;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
canvasContainer.appendChild(renderer.domElement);

// Glowing Star Texture
function createStarTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    
    ctx.beginPath();
    const cx = 32;
    const cy = 32;
    const spikes = 5;
    const outerRadius = 30;
    const innerRadius = 12;
    let rot = Math.PI / 2 * 3;
    const step = Math.PI / spikes;

    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        const x = cx + Math.cos(rot) * outerRadius;
        const y = cy + Math.sin(rot) * outerRadius;
        ctx.lineTo(x, y);
        rot += step;

        const x2 = cx + Math.cos(rot) * innerRadius;
        const y2 = cy + Math.sin(rot) * innerRadius;
        ctx.lineTo(x2, y2);
        rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius);
    ctx.closePath();
    
    ctx.shadowColor = 'white';
    ctx.shadowBlur = 10;
    ctx.fillStyle = 'white';
    ctx.fill();
    
    return new THREE.CanvasTexture(canvas);
}

// Particle System
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(NUM_PARTICLES * 3);
const velocities = [];
const colors = new Float32Array(NUM_PARTICLES * 3);
const particleBases = [];

// Screen dimensions in world space
const vFov = camera.fov * Math.PI / 180;
let planeHeight = 2 * Math.tan(vFov / 2) * camera.position.z;
let planeWidth = planeHeight * (window.innerWidth / window.innerHeight);

for (let i = 0; i < NUM_PARTICLES; i++) {
    const x = (Math.random() - 0.5) * planeWidth * 4.0;
    const y = (Math.random() - 0.5) * planeHeight * 4.0;
    const z = (Math.random() - 0.5) * 50;
    
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    
    velocities.push(new THREE.Vector3(0, 0, 0));
    
    // Assign random properties to each particle for organic movement
    particleBases.push({
        origin: new THREE.Vector3(x, y, z),
        speed: Math.random() * 0.05 + 0.02,
        wander: Math.random() * Math.PI * 2,
        targetTip: Math.floor(Math.random() * 5) // Which fingertip to prefer
    });

    PARTICLE_COLOR.toArray(colors, i * 3);
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const material = new THREE.PointsMaterial({
    size: 15.0,
    map: createStarTexture(),
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    vertexColors: true,
    opacity: 0.8
});

const particles = new THREE.Points(geometry, material);
scene.add(particles);

// --- MEDIAPIPE SETUP ---
let handLandmarker = null;
let lastVideoTime = -1;
let isTracking = false;

// Shared state
let handActive = false;
let isFist = false;
let handCenter = new THREE.Vector3();
let fingertips = []; // Array of THREE.Vector3
let handVelocity = new THREE.Vector3();
let smoothedHandVelocity = new THREE.Vector3();
let lastHandCenter = new THREE.Vector3();
let handQuaternion = new THREE.Quaternion();
const identityQuaternion = new THREE.Quaternion();

async function initMediaPipe() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        
        statusText.innerText = "SYSTEM ACTIVE";
        statusIndicator.classList.add('active');
        
        startWebcam();
    } catch (e) {
        console.error(e);
        statusText.innerText = "SYS ERROR: " + e.message;
    }
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            isTracking = true;
            animate();
        });
    } catch (err) {
        console.error("Camera error:", err);
        statusText.innerText = "CAMERA OFFLINE";
    }
}

// Convert MediaPipe landmark to Three.js world position
function getPos(landmark) {
    return new THREE.Vector3(
        (landmark.x - 0.5) * planeWidth,
        -(landmark.y - 0.5) * planeHeight,
        -landmark.z * planeWidth * 0.5 // Scale Z appropriately
    );
}

function checkFist(landmarks) {
    const wrist = landmarks[0];
    const fingersCurled = [
        [8, 5],   // Index
        [12, 9],  // Middle
        [16, 13], // Ring
        [20, 17]  // Pinky
    ].map(pair => {
        const tip = landmarks[pair[0]];
        const mcp = landmarks[pair[1]];
        const tipDist = Math.hypot(tip.x - wrist.x, tip.y - wrist.y, tip.z - wrist.z);
        const mcpDist = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z);
        return tipDist < mcpDist;
    });
    return fingersCurled.every(curled => curled);
}

function predictWebcam() {
    if (!handLandmarker || !video.currentTime) return;
    
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, performance.now());
        
        if (results.landmarks && results.landmarks.length > 0) {
            const lm = results.landmarks[0];
            const wasActive = handActive;
            handActive = true;
            
            // Get fingertips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
            fingertips = [
                getPos(lm[4]),
                getPos(lm[8]),
                getPos(lm[12]),
                getPos(lm[16]),
                getPos(lm[20])
            ];
            
            // Hand center (average of all landmarks)
            const currentHandCenter = new THREE.Vector3();
            lm.forEach(l => currentHandCenter.add(getPos(l)));
            currentHandCenter.divideScalar(lm.length);
            
            if (wasActive) {
                const delta = currentHandCenter.clone().sub(lastHandCenter);
                handVelocity.copy(delta).multiplyScalar(30.0);
            } else {
                handVelocity.set(0, 0, 0);
            }
            
            handCenter.copy(currentHandCenter);
            lastHandCenter.copy(currentHandCenter);
            
            // Compute 3D rotation
            const wrist = getPos(lm[0]);
            const indexMCP = getPos(lm[5]);
            const pinkyMCP = getPos(lm[17]);
            const middleMCP = getPos(lm[9]);

            const upDir = middleMCP.clone().sub(wrist).normalize();
            const rightDir = pinkyMCP.clone().sub(indexMCP).normalize();
            const forwardDir = new THREE.Vector3().crossVectors(rightDir, upDir).normalize();
            rightDir.crossVectors(upDir, forwardDir).normalize();
            
            const matrix = new THREE.Matrix4().makeBasis(rightDir, upDir, forwardDir);
            handQuaternion.setFromRotationMatrix(matrix);
            
            isFist = checkFist(lm);
            handStateText.innerText = isFist ? "CLENCHED (SCATTER)" : "OPEN (FLOCKING)";
            handStateText.style.color = isFist ? "var(--secondary-color)" : "var(--primary-color)";
        } else {
            handActive = false;
            handStateText.innerText = "SCANNING";
            handStateText.style.color = "rgba(255, 255, 255, 0.6)";
            handVelocity.set(0, 0, 0);
        }
    }
}

// --- ANIMATION LOOP ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    
    if (isTracking) {
        predictWebcam();
    }
    
    const dt = Math.min(clock.getDelta(), 0.1);
    const time = clock.getElapsedTime();
    
    smoothedHandVelocity.lerp(handVelocity, dt * 5.0);
    
    if (handActive) {
        particles.quaternion.slerp(handQuaternion, dt * 5.0);
    } else {
        particles.quaternion.slerp(identityQuaternion, dt * 2.0);
        handVelocity.lerp(new THREE.Vector3(0,0,0), dt * 5.0);
    }
    
    const positionsAttr = geometry.attributes.position;
    const colorsAttr = geometry.attributes.color;
    
    const lerpFactor = dt * 5.0;
    
    for (let i = 0; i < NUM_PARTICLES; i++) {
        const i3 = i * 3;
        const pos = new THREE.Vector3(positionsAttr.array[i3], positionsAttr.array[i3+1], positionsAttr.array[i3+2]);
        const vel = velocities[i];
        const base = particleBases[i];
        
        let force = new THREE.Vector3(0, 0, 0);
        
        // Spring back to origin (shifted by hand velocity to empty the screen dynamically)
        let springStrength = handActive ? 0.5 : 3.0; // Weaker spring when interacting
        const targetOrigin = base.origin.clone().add(smoothedHandVelocity.clone().multiplyScalar(15.0));
        const springForce = targetOrigin.sub(pos).multiplyScalar(springStrength);
        force.add(springForce);
        
        if (handActive) {
            if (isFist) {
                // Scatter away from hand center powerfully
                const dir = pos.clone().sub(handCenter);
                const dist = dir.length();
                if (dist > 0.1 && dist < 50.0) {
                    force.add(dir.normalize().multiplyScalar(2000 / (dist + 1)));
                }
            } else {
                // Flock to fingertips, but do not group
                const target = fingertips[base.targetTip];
                const dir = target.clone().sub(pos);
                const dist = dir.length();
                
                if (dist > 10.0) {
                    // Attraction force
                    force.add(dir.normalize().multiplyScalar(dist * 4.0));
                } else {
                    // Strong repulsion to prevent clumping
                    force.add(dir.normalize().multiplyScalar(-300.0 / (dist + 0.1)));
                }
                
                // Add some noise/swirl around the fingertip
                const swirl = new THREE.Vector3(
                    Math.sin(time * 3 + i) * 10,
                    Math.cos(time * 3.5 + i) * 10,
                    Math.sin(time * 2.5 + i) * 10
                );
                force.add(swirl);
            }
        } else {
            // Idle wander while returning to full screen formation
            base.wander += (Math.random() - 0.5) * 0.5;
            force.add(new THREE.Vector3(
                Math.cos(base.wander) * 5,
                Math.sin(base.wander) * 5,
                Math.sin(base.wander * 0.5) * 5
            ));
        }
        
        // Update velocity
        vel.add(force.multiplyScalar(dt));
        
        // Friction/damping
        const damping = isFist ? 0.95 : 0.9;
        vel.multiplyScalar(damping);
        
        // Speed limit
        if (vel.length() > 50) {
            vel.normalize().multiplyScalar(50);
        }
        
        // Update position
        pos.add(vel.clone().multiplyScalar(dt));
        
        // Write back position
        positionsAttr.array[i3] = pos.x;
        positionsAttr.array[i3+1] = pos.y;
        positionsAttr.array[i3+2] = pos.z;
        
        // Target color logic based on velocity direction
        let particleTargetColor = PARTICLE_COLOR;
        
        if (isFist) {
            particleTargetColor = SCATTER_COLOR;
        } else if (vel.length() > 0.5) {
            if (Math.abs(vel.x) > Math.abs(vel.y)) {
                // Horizontal dominance
                particleTargetColor = vel.x > 0 ? COLOR_YELLOW : COLOR_BLUE;
            } else {
                // Vertical dominance
                particleTargetColor = vel.y > 0 ? COLOR_ORANGE : COLOR_PURPLE;
            }
        }
        
        // Update color
        const currentColor = new THREE.Color(colorsAttr.array[i3], colorsAttr.array[i3+1], colorsAttr.array[i3+2]);
        currentColor.lerp(particleTargetColor, lerpFactor);
        
        // Add random flickering
        const flicker = 0.8 + Math.random() * 0.4;
        colorsAttr.array[i3] = Math.min(1.0, currentColor.r * flicker);
        colorsAttr.array[i3+1] = Math.min(1.0, currentColor.g * flicker);
        colorsAttr.array[i3+2] = Math.min(1.0, currentColor.b * flicker);
    }
    
    positionsAttr.needsUpdate = true;
    colorsAttr.needsUpdate = true;
    
    // Slight camera rotation for parallax
    if (handActive && !isFist) {
        const targetX = (handCenter.x / planeWidth) * 5;
        const targetY = (handCenter.y / planeHeight) * 5;
        camera.position.x += (targetX - camera.position.x) * 0.05;
        camera.position.y += (targetY - camera.position.y) * 0.05;
        camera.lookAt(0, 0, 0);
    } else {
        camera.position.x += (0 - camera.position.x) * 0.05;
        camera.position.y += (0 - camera.position.y) * 0.05;
        camera.lookAt(0, 0, 0);
    }
    
    renderer.render(scene, camera);
}

// Handle resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    // Update plane dimensions
    const newVFov = camera.fov * Math.PI / 180;
    planeHeight = 2 * Math.tan(newVFov / 2) * camera.position.z;
    planeWidth = planeHeight * (window.innerWidth / window.innerHeight);
});

// Start MediaPipe
initMediaPipe();
