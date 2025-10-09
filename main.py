"""
VoiceFlow Tools - Voice Analysis API
FastAPI backend for acoustic voice analysis
Author: Dr. Jorge C. Lucero
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import io
import base64
from typing import Dict, Any, Optional
import tempfile
import os
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="VoiceFlow Voice Analyzer API",
    description="Free acoustic voice analysis for clinicians",
    version="1.0.0"
)

# Configure CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    """Response model for voice analysis"""
    f0: Dict[str, float]
    jitter: Dict[str, float]
    shimmer: Dict[str, float]
    hnr: Dict[str, float]
    interpretation: Dict[str, Any]
    metadata: Dict[str, Any]

@app.get("/")
async def root():
    """API health check and info"""
    return {
        "status": "active",
        "service": "VoiceFlow Voice Analyzer",
        "version": "1.0.0",
        "author": "Dr. Jorge C. Lucero",
        "description": "Free acoustic voice analysis API"
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze voice recording and return acoustic parameters
    """
    import logging
    import traceback
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate file
    if not file.content_type.startswith('audio'):
        raise HTTPException(status_code=400, detail="Please upload an audio file")
    
    # Size limit (50MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 50MB")
    
    tmp_path = None
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Saved to temp file: {tmp_path}")
        
        # Load audio with librosa for basic processing
        logger.info("Loading with librosa...")
        audio_data, sample_rate = librosa.load(tmp_path, sr=None)
        duration = len(audio_data) / sample_rate
        
        logger.info("Loading with Parselmouth...")
        # Load with Parselmouth for acoustic analysis
        sound = parselmouth.Sound(tmp_path)
        
        logger.info("Analyzing F0...")
        f0_data = analyze_f0(sound)
        
        logger.info("Analyzing jitter...")
        jitter_data = analyze_jitter(sound)
        
        logger.info("Analyzing shimmer...")
        shimmer_data = analyze_shimmer(sound)
        
        logger.info("Analyzing HNR...")
        hnr_data = analyze_hnr(sound)
        
        logger.info("Generating interpretation...")
        interpretation = generate_interpretation(f0_data, jitter_data, shimmer_data, hnr_data)
        
        # Clean up temp file
        os.unlink(tmp_path)
        logger.info("Analysis completed successfully")
        
        return AnalysisResult(
            f0=f0_data,
            jitter=jitter_data,
            shimmer=shimmer_data,
            hnr=hnr_data,
            interpretation=interpretation,
            metadata={
                "filename": file.filename,
                "duration": round(duration, 2),
                "sample_rate": int(sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Clean up temp file if it exists
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    
def analyze_f0(sound: parselmouth.Sound) -> Dict[str, float]:
    """Extract fundamental frequency statistics"""
    try:
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # Remove unvoiced segments
        
        if len(f0_values) == 0:
            return {
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "status": "Could not detect pitch"
            }
        
        return {
            "mean": float(np.mean(f0_values)),
            "std": float(np.std(f0_values)),
            "min": float(np.min(f0_values)),
            "max": float(np.max(f0_values)),
            "percentile_5": float(np.percentile(f0_values, 5)),
            "percentile_95": float(np.percentile(f0_values, 95)),
            "status": get_f0_status(float(np.mean(f0_values)))
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_jitter(sound: parselmouth.Sound) -> Dict[str, float]:
    """Calculate jitter parameters"""
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        # Local jitter
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Jitter percent
        jitter_percent = local_jitter * 100
        
        # RAP
        rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        return {
            "local": float(local_jitter),
            "percent": float(jitter_percent),
            "rap": float(rap),
            "status": get_jitter_status(jitter_percent)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_shimmer(sound: parselmouth.Sound) -> Dict[str, float]:
    """Calculate shimmer parameters"""
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        # Local shimmer
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 
                            0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Shimmer percent
        shimmer_percent = local_shimmer * 100
        
        # APQ3
        apq3 = call([sound, point_process], "Get shimmer (apq3)", 
                   0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        return {
            "local": float(local_shimmer),
            "percent": float(shimmer_percent),
            "apq3": float(apq3),
            "status": get_shimmer_status(shimmer_percent)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_hnr(sound: parselmouth.Sound) -> Dict[str, float]:
    """Calculate Harmonics-to-Noise Ratio"""
    try:
        harmonicity = sound.to_harmonicity()
        hnr_values = harmonicity.values[harmonicity.values != -200]
        
        if len(hnr_values) == 0:
            return {
                "mean": 0,
                "std": 0,
                "status": "Could not calculate HNR"
            }
        
        mean_hnr = float(np.mean(hnr_values))
        
        return {
            "mean": mean_hnr,
            "std": float(np.std(hnr_values)),
            "min": float(np.min(hnr_values)),
            "max": float(np.max(hnr_values)),
            "status": get_hnr_status(mean_hnr)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def get_f0_status(f0_mean: float) -> str:
    """Interpret F0 value"""
    if f0_mean < 85:
        return "Very low - possible pathology"
    elif 85 <= f0_mean < 155:
        return "Normal range (adult male)"
    elif 155 <= f0_mean < 255:
        return "Normal range (adult female)"
    else:
        return "High - possible tension or pediatric"

def get_jitter_status(jitter_percent: float) -> str:
    """Interpret jitter value"""
    if jitter_percent < 1.0:
        return "Normal"
    elif jitter_percent < 2.0:
        return "Slightly elevated"
    else:
        return "Elevated - suggests voice disorder"

def get_shimmer_status(shimmer_percent: float) -> str:
    """Interpret shimmer value"""
    if shimmer_percent < 3.0:
        return "Normal"
    elif shimmer_percent < 5.0:
        return "Slightly elevated"
    else:
        return "Elevated - suggests voice disorder"

def get_hnr_status(hnr_mean: float) -> str:
    """Interpret HNR value"""
    if hnr_mean > 20:
        return "Good voice quality"
    elif hnr_mean > 15:
        return "Mild noise component"
    else:
        return "Significant noise - possible disorder"

def generate_interpretation(f0: Dict, jitter: Dict, shimmer: Dict, hnr: Dict) -> Dict:
    """Generate clinical interpretation of results"""
    
    concerns = []
    recommendations = []
    
    # Check each parameter
    if 'percent' in jitter and jitter['percent'] > 1.0:
        concerns.append("Elevated jitter (pitch instability)")
        recommendations.append("Consider vocal fold examination")
    
    if 'percent' in shimmer and shimmer['percent'] > 3.0:
        concerns.append("Elevated shimmer (amplitude instability)")
        recommendations.append("Assess for vocal fold lesions")
    
    if 'mean' in hnr and hnr['mean'] < 20:
        concerns.append("Reduced harmonics-to-noise ratio")
        recommendations.append("Evaluate for breathiness or roughness")
    
    if 'mean' in f0:
        if f0['mean'] < 85 or f0['mean'] > 300:
            concerns.append("Abnormal fundamental frequency")
            recommendations.append("Check for structural or functional issues")
    
    # Overall assessment
    if len(concerns) == 0:
        overall = "Voice parameters within normal limits"
        severity = "normal"
    elif len(concerns) == 1:
        overall = "Mild voice quality concerns noted"
        severity = "mild"
    elif len(concerns) == 2:
        overall = "Moderate voice disorder indicators present"
        severity = "moderate"
    else:
        overall = "Multiple parameters suggest significant voice disorder"
        severity = "severe"
    
    return {
        "overall_assessment": overall,
        "severity": severity,
        "concerns": concerns,
        "recommendations": recommendations,
        "clinical_action": get_clinical_action(severity)
    }

def get_clinical_action(severity: str) -> str:
    """Suggest clinical action based on severity"""
    actions = {
        "normal": "Continue routine monitoring",
        "mild": "Monitor and consider voice hygiene education",
        "moderate": "Recommend comprehensive voice evaluation",
        "severe": "Urgent referral to otolaryngology recommended"
    }
    return actions.get(severity, "Clinical correlation recommended")

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "voice-analyzer"}

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)