"""
VoiceFlow Tools - Voice Analysis API
FastAPI backend for acoustic voice analysis
Author: Dr. Jorge C. Lucero
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import parselmouth
from parselmouth.praat import call
import io
import base64
from typing import Dict, Any, Optional, Union, List
import tempfile
import os
from pydantic import BaseModel
from datetime import datetime
import logging
import traceback

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    f0: Dict[str, Union[float, str]]
    jitter: Dict[str, Union[float, str]]
    shimmer: Dict[str, Union[float, str]]
    hnr: Dict[str, Union[float, str]]
    interpretation: Dict[str, Any]
    metadata: Dict[str, Any]

class FeedbackSubmission(BaseModel):
    """Model for user feedback"""
    rating: int
    comment: Optional[str] = None
    feature_request: Optional[str] = None
    user_type: Optional[str] = None
    timestamp: Optional[str] = None

class PDFRequest(BaseModel):
    """Model for PDF generation request"""
    analysis_results: AnalysisResult
    patient_info: Optional[Dict[str, str]] = None

# In-memory feedback storage (replace with database in production)
feedback_storage: List[Dict] = []

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
    
    Parameters:
    - file: Audio file (WAV, MP3, M4A)
    
    Returns:
    - JSON with F0, jitter, shimmer, HNR, and clinical interpretation
    """
    
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
        
        # Load with Parselmouth for acoustic analysis (much faster than librosa)
        logger.info("Loading audio with Parselmouth...")
        sound = parselmouth.Sound(tmp_path)
        
        # Get duration and sample rate from Parselmouth
        duration = sound.duration
        sample_rate = int(sound.sampling_frequency)
        
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
                "sample_rate": int(sample_rate),
                "analysis_date": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Clean up temp file if it exists
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/generate-pdf")
async def generate_pdf_report(request: PDFRequest):
    """
    Generate PDF report from analysis results
    
    Parameters:
    - request: PDFRequest with analysis results and optional patient info
    
    Returns:
    - PDF file as downloadable stream
    """
    try:
        logger.info("Generating PDF report...")
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Container for PDF elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("Voice Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        metadata = request.analysis_results.metadata
        elements.append(Paragraph("Recording Information", heading_style))
        
        info_data = [
            ["Filename:", metadata.get('filename', 'N/A')],
            ["Duration:", f"{metadata.get('duration', 0)} seconds"],
            ["Sample Rate:", f"{metadata.get('sample_rate', 0)} Hz"],
            ["Analysis Date:", metadata.get('analysis_date', datetime.now().isoformat())[:19].replace('T', ' ')],
        ]
        
        # Add patient info if provided
        if request.patient_info:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("Patient Information", heading_style))
            patient_data = [[k.replace('_', ' ').title() + ':', v] for k, v in request.patient_info.items()]
            info_data.extend(patient_data)
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Acoustic Parameters
        elements.append(Paragraph("Acoustic Measurements", heading_style))
        
        results = request.analysis_results
        
        # F0
        f0_data = [
            ["Parameter", "Value", "Status"],
            ["Mean F0", f"{results.f0.get('mean', 0):.1f} Hz", results.f0.get('status', 'N/A')],
            ["F0 Range", f"{results.f0.get('min', 0):.1f} - {results.f0.get('max', 0):.1f} Hz", ""],
            ["F0 Std Dev", f"{results.f0.get('std', 0):.1f} Hz", ""],
        ]
        
        # Jitter
        jitter_data = [
            ["Jitter (Local)", f"{results.jitter.get('local', 0):.4f}", results.jitter.get('status', 'N/A')],
            ["Jitter (%)", f"{results.jitter.get('percent', 0):.2f}%", ""],
            ["RAP", f"{results.jitter.get('rap', 0):.4f}", ""],
        ]
        
        # Shimmer
        shimmer_data = [
            ["Shimmer (Local)", f"{results.shimmer.get('local', 0):.4f}", results.shimmer.get('status', 'N/A')],
            ["Shimmer (%)", f"{results.shimmer.get('percent', 0):.2f}%", ""],
            ["APQ3", f"{results.shimmer.get('apq3', 0):.4f}", ""],
        ]
        
        # HNR
        hnr_data = [
            ["Mean HNR", f"{results.hnr.get('mean', 0):.1f} dB", results.hnr.get('status', 'N/A')],
            ["HNR Range", f"{results.hnr.get('min', 0):.1f} - {results.hnr.get('max', 0):.1f} dB", ""],
        ]
        
        # Combine all measurements
        all_data = [["Parameter", "Value", "Status"]] + f0_data[1:] + jitter_data + shimmer_data + hnr_data
        
        measurements_table = Table(all_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        measurements_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(measurements_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Clinical Interpretation
        elements.append(Paragraph("Clinical Interpretation", heading_style))
        
        interpretation = results.interpretation
        elements.append(Paragraph(
            f"<b>Overall Assessment:</b> {interpretation.get('overall_assessment', 'N/A')}", 
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph(
            f"<b>Severity:</b> {interpretation.get('severity', 'N/A').upper()}", 
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Concerns
        if interpretation.get('concerns'):
            elements.append(Paragraph("<b>Clinical Concerns:</b>", styles['Normal']))
            for concern in interpretation['concerns']:
                elements.append(Paragraph(f"• {concern}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        if interpretation.get('recommendations'):
            elements.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
            for rec in interpretation['recommendations']:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Clinical Action
        elements.append(Paragraph(
            f"<b>Suggested Action:</b> {interpretation.get('clinical_action', 'N/A')}", 
            styles['Normal']
        ))
        
        # Footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(
            "<i>This report was generated by VoiceFlow Tools - Free Voice Analysis for Clinicians</i>",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        ))
        elements.append(Paragraph(
            "<i>Created by Dr. Jorge C. Lucero | For clinical correlation and professional use only</i>",
            ParagraphStyle('Footer2', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        logger.info("PDF generated successfully")
        
        # Return as downloadable file
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=voice_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    Submit user feedback
    
    Parameters:
    - feedback: FeedbackSubmission with rating and optional comments
    
    Returns:
    - Confirmation message
    """
    try:
        # Add timestamp if not provided
        if not feedback.timestamp:
            feedback.timestamp = datetime.now().isoformat()
        
        # Store feedback (in production, save to database)
        feedback_data = feedback.dict()
        feedback_storage.append(feedback_data)
        
        logger.info(f"Feedback received: Rating {feedback.rating}/5")
        if feedback.comment:
            logger.info(f"Comment: {feedback.comment}")
        if feedback.feature_request:
            logger.info(f"Feature request: {feedback.feature_request}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "feedback_id": len(feedback_storage)
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get feedback statistics (for admin use)
    
    Returns:
    - Summary of feedback data
    """
    if not feedback_storage:
        return {
            "total_feedback": 0,
            "average_rating": 0,
            "ratings_distribution": {}
        }
    
    ratings = [f['rating'] for f in feedback_storage]
    
    return {
        "total_feedback": len(feedback_storage),
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "ratings_distribution": {
            str(i): ratings.count(i) for i in range(1, 6)
        },
        "recent_comments": [
            {
                "rating": f['rating'],
                "comment": f.get('comment', ''),
                "timestamp": f.get('timestamp', '')
            }
            for f in feedback_storage[-10:]  # Last 10 feedbacks
        ]
    }

# Analysis helper functions (keep existing code)
def analyze_f0(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Extract fundamental frequency statistics"""
    try:
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        
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

def analyze_jitter(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Calculate jitter parameters"""
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_percent = local_jitter * 100
        rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        return {
            "local": float(local_jitter),
            "percent": float(jitter_percent),
            "rap": float(rap),
            "status": get_jitter_status(jitter_percent)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_shimmer(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Calculate shimmer parameters"""
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 
                            0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_percent = local_shimmer * 100
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

def analyze_hnr(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
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

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "voice-analyzer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
