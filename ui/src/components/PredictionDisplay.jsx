import React, { useEffect, useState } from "react";

// convertir le masque en image PNG base64
function maskArrayToImage(maskArray) {
  if (!maskArray) return null;
  const height = maskArray.length;
  const width = maskArray[0].length;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const pixel = maskArray[y][x];
      imageData.data[idx] = Math.round(pixel[0] * 255);     // R
      imageData.data[idx + 1] = Math.round(pixel[1] * 255); // G
      imageData.data[idx + 2] = Math.round(pixel[2] * 255); // B
      imageData.data[idx + 3] = 255;                        // A
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

const PredictionDisplay = ({ predMask, classification }) => {
  const [maskUrl, setMaskUrl] = useState(null);

  useEffect(() => {
    if (predMask && Array.isArray(predMask)) {
      console.log(
        "predMask shape:",
        predMask.length,
        predMask[0]?.length,
        predMask[0]?.[0]?.length,
        Array.isArray(predMask[0][0][0]) ? "batch dimension detected" : "no batch"
      );
      if (Array.isArray(predMask[0][0][0])) {
        setMaskUrl(maskArrayToImage(predMask[0]));
      } else {
        setMaskUrl(maskArrayToImage(predMask));
      }
    } else {
      setMaskUrl(null);
    }
  }, [predMask]);

  if (!predMask) return null;

  return (
    <div className="prediction-card">
      <h3>Masques prédits :</h3>
      <div className="mask-container">
        {maskUrl && (
          <img src={maskUrl} alt="Masque de prédiction" style={{ maxWidth: '400px' }} />
        )}
      </div>
      <h3 className="classification-text">Résultat : {classification}</h3>
    </div>
  );
};

export default PredictionDisplay;