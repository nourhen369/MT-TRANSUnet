import React, { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight, faUpload } from "@fortawesome/free-solid-svg-icons";

const ImageUploader = ({ handleFileChange, handleSubmit, loading }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  return (
    <div
      className={`uploader-card ${dragOver ? "drag-over" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={(e) => {
        e.preventDefault();
        handleFileChange({ target: { files: e.dataTransfer.files } });
        setDragOver(false);
      }}
    >
      <h2>Déposez une image d’embryon ici</h2>
      <FontAwesomeIcon icon={faUpload} size="3x" style={{ margin: "1rem 0" }} />
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={loading} className="app-button">
        {loading ? "Prédiction en cours..." : "Lancer la prédiction"}
        <FontAwesomeIcon icon={faArrowRight} style={{ marginLeft: "8px" }} />
      </button>
    </div>
  );
};

export default ImageUploader;