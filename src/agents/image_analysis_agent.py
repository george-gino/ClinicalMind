from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClinicalMind Image Analysis Agent")

class ChestXRayAnalyzer:
    """Simple class for analyzing chest X-rays using basic image processing."""
    
    def __init__(self):
        self.target_size = (224, 224)
        # Define the conditions we'll detect
        self.conditions = [
            "Normal", 
            "Potential Opacity", 
            "Potential Pneumonia", 
            "Potential Pleural Effusion"
        ]
        
    def _extract_image_features(self, img_array):
        """Extract basic image features for analysis."""
        features = {}
        
        # Convert to grayscale if it's RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Calculate basic statistics
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        
        # Edge detection (simplified)
        try:
            from skimage import feature as skfeature
            edges = skfeature.canny(gray, sigma=1)
            features['edge_density'] = np.sum(edges) / edges.size
        except:
            features['edge_density'] = 0.1  # fallback value
        
        # Region analysis (simplified)
        bright_regions = gray > 0.7
        dark_regions = gray < 0.3
        features['bright_region_ratio'] = np.sum(bright_regions) / bright_regions.size
        features['dark_region_ratio'] = np.sum(dark_regions) / dark_regions.size
        
        return features
    
    def analyze_xray(self, img_array):
        """Analyze a chest X-ray image and detect potential conditions."""
        # Extract image features
        features = self._extract_image_features(img_array)
        
        # Simulated detection logic based on image features
        # This is a simplified approach for educational purposes
        
        findings = []
        confidence_scores = []
        
        # Simple rule-based detection (for demonstration)
        if features['mean_intensity'] > 0.6 and features['edge_density'] < 0.05:
            # Bright image with few edges - might be overexposed, but we'll call it normal
            findings.append("Normal")
            confidence_scores.append(0.8)
        
        if 0.05 < features['edge_density'] < 0.15 and features['dark_region_ratio'] > 0.2:
            # More edges and dark regions could suggest opacity
            findings.append("Potential Opacity")
            confidence_scores.append(0.6)
        
        if features['edge_density'] > 0.1 and features['dark_region_ratio'] > 0.3:
            # Higher edge density with significant dark regions might suggest pneumonia
            findings.append("Potential Pneumonia")
            confidence_scores.append(0.5)
        
        if features['bright_region_ratio'] > 0.4 and features['edge_density'] < 0.08:
            # Bright regions with some edges might suggest pleural effusion
            findings.append("Potential Pleural Effusion")
            confidence_scores.append(0.4)
        
        # If nothing was detected, default to normal
        if not findings:
            findings.append("Normal")
            confidence_scores.append(0.7)
        
        # Package the results
        results = {
            "primary_finding": findings[0],
            "confidence": confidence_scores[0],
            "all_findings": [
                {"condition": finding, "confidence": score} 
                for finding, score in zip(findings, confidence_scores)
            ],
            "image_features": features
        }
        
        return results

class ImageAnalyzer:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.dicom', '.dcm']
        
        # Initialize the deep learning model if available
        model_path = '/app/data/models/chest_xray_model.h5'
        local_model_path = 'data/models/chest_xray_model.h5'
        
        # Check both possible paths
        if os.path.exists(model_path):
            logger.info(f"Found model at Docker path: {model_path}")
            self.model_file_path = model_path
        elif os.path.exists(local_model_path):
            logger.info(f"Found model at local path: {local_model_path}")
            self.model_file_path = local_model_path
        else:
            logger.warning(f"No model file found at {model_path} or {local_model_path}")
            self.model_file_path = None
        
        # Try to load the deep learning model
        self.dl_model_available = False
        if self.model_file_path:
            try:
                from src.models.deep_learning_model import ChestXRayModel
                self.chest_xray_model = ChestXRayModel()
                self.dl_model_available = True
                logger.info("Deep learning model initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize deep learning model: {e}")
                self.dl_model_available = False
        
        # Always initialize the rule-based analyzer as fallback
        self.xray_analyzer = ChestXRayAnalyzer()
    
    async def process_image(self, image_data: bytes):
        """Process image bytes and extract detailed properties."""
        try:
            # Load the image
            image = Image.open(io.BytesIO(image_data))
            
            # Extract basic properties
            image_properties = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
            }
            
            # Get histogram data (distribution of pixel values)
            if image.mode == "RGB":
                r, g, b = image.convert("RGB").split()
                r_hist = r.histogram()
                g_hist = g.histogram()
                b_hist = b.histogram()
                
                # Calculate average brightness
                r_brightness = sum(i * v for i, v in enumerate(r_hist)) / sum(r_hist) if sum(r_hist) > 0 else 0
                g_brightness = sum(i * v for i, v in enumerate(g_hist)) / sum(g_hist) if sum(g_hist) > 0 else 0
                b_brightness = sum(i * v for i, v in enumerate(b_hist)) / sum(b_hist) if sum(b_hist) > 0 else 0
                
                avg_brightness = (r_brightness + g_brightness + b_brightness) / 3 / 255
                image_properties["brightness"] = round(avg_brightness, 2)
            else:
                # For grayscale images
                hist = image.histogram()
                brightness = sum(i * v for i, v in enumerate(hist)) / sum(hist) if sum(hist) > 0 else 0
                image_properties["brightness"] = round(brightness / 255, 2)
            
            # Resize image for consistency
            processed_image = image.resize((224, 224))
            
            # Convert to numpy array for potential future model use
            img_array = np.array(processed_image) / 255.0
            
            # Check if it's likely a medical image
            is_medical = self._is_likely_medical_image(img_array, image_properties)
            
            # Basic image quality assessment
            quality = self._assess_image_quality(img_array, image_properties)
            
            return {
                "status": "success",
                "message": "Image processed successfully",
                "image_properties": image_properties,
                "analysis": {
                    "is_medical_image": is_medical,
                    "image_quality": quality,
                    "abnormality_detected": False,  # Will be updated if we run diagnosis
                    "confidence": 0.0               # Will be updated if we run diagnosis
                }
            }
        except Exception as e:
            import traceback
            return {
                "status": "error", 
                "message": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _is_likely_medical_image(self, img_array, properties):
        """Simple heuristic to guess if an image might be a medical image."""
        # Medical images often have:
        # 1. Specific aspect ratios
        # 2. Limited color palette
        # 3. Certain brightness profiles
        
        # Check if grayscale or limited color palette
        if properties["mode"] in ["L", "1"]:
            return True
        
        # Check brightness (medical images often have specific brightness patterns)
        if 0.2 <= properties["brightness"] <= 0.6:
            return True
        
        return False
    
    def _assess_image_quality(self, img_array, properties):
        """Basic assessment of image quality."""
        # Simple quality metrics
        quality_score = 0.5  # Neutral starting point
        
        # Higher resolution generally means better quality
        if properties["width"] >= 1000 or properties["height"] >= 1000:
            quality_score += 0.2
        
        # Very small images might be poor quality
        if properties["width"] < 200 or properties["height"] < 200:
            quality_score -= 0.2
        
        # Extremely bright or dark images might be poor quality
        if properties["brightness"] < 0.1 or properties["brightness"] > 0.9:
            quality_score -= 0.2
        
        # Clamp to 0-1 range
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Map to descriptive text
        if quality_score >= 0.8:
            return "high"
        elif quality_score >= 0.5:
            return "acceptable"
        else:
            return "low"
    
    async def analyze_chest_xray(self, image_data: bytes):
        """Perform specific analysis for chest X-rays."""
        # First get basic image processing
        base_results = await self.process_image(image_data)
        
        if base_results["status"] != "success":
            return base_results
        
        # Load and resize the image for analysis
        image = Image.open(io.BytesIO(image_data))
        processed_image = image.resize((224, 224))
        img_array = np.array(processed_image) / 255.0
        
        # Try deep learning model if available, otherwise use rule-based
        if hasattr(self, 'dl_model_available') and self.dl_model_available:
            try:
                logger.info("Attempting to use deep learning model for analysis")
                xray_results = self.chest_xray_model.predict(img_array)
                model_type = "deep learning"
                logger.info("Used deep learning model for analysis")
            except Exception as e:
                logger.warning(f"Error using deep learning model: {e}, falling back to rule-based")
                logger.warning(traceback.format_exc())
                xray_results = self.xray_analyzer.analyze_xray(img_array)
                model_type = "rule-based (fallback)"
        else:
            logger.info("Using rule-based analysis (no deep learning model available)")
            xray_results = self.xray_analyzer.analyze_xray(img_array)
            model_type = "rule-based"
        
        # Update the base results with diagnostic information
        base_results["analysis"]["abnormality_detected"] = xray_results["primary_finding"] != "Normal"
        base_results["analysis"]["confidence"] = xray_results["confidence"]
        
        # Add the full diagnostic results
        base_results["diagnostic_results"] = {
            "primary_finding": xray_results["primary_finding"],
            "confidence": xray_results["confidence"],
            "all_findings": xray_results["all_findings"],
            "model_type": model_type,
            "disclaimer": "EDUCATIONAL DEMONSTRATION ONLY - NOT FOR CLINICAL USE",
            "explanation": "This analysis uses simplified techniques for educational purposes. In a real system, validated deep learning models trained on large datasets would be used."
        }
        
        return base_results

# Initialize the analyzer
analyzer = ImageAnalyzer()

@app.get("/")
async def root():
    return {"message": "ClinicalMind Image Analysis Agent API"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a medical image uploaded by the user."""
    image_data = await file.read()
    results = await analyzer.process_image(image_data)
    return results

@app.post("/analyze-chest-xray")
async def analyze_chest_xray(file: UploadFile = File(...)):
    """Analyze a chest X-ray with diagnostic capabilities."""
    image_data = await file.read()
    results = await analyzer.analyze_chest_xray(image_data)
    return results

@app.get("/health")
async def health_check():
    return {"status": "healthy"}