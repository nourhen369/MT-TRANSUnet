import React from "react";

const PredictionDisplay = ({ predMask, classification }) => {
  if (!predMask) return null;

  return (
    <div className="prediction-card">
      <h3>Masques prédits :</h3>
      <div className="mask-container">
        <img src={predMask} alt="Masque de prédiction" style={{ maxWidth: '400px' }} />
      </div>
      <h3 className="classification-text">Classification : {classification}</h3>
    </div>
  );
};

export default PredictionDisplay;