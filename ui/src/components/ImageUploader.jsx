import React, { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight, faUpload } from "@fortawesome/free-solid-svg-icons";

const ImageUploader = ({ handleFileChange, handleSubmit, loading }) => {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const onFileChange = (e) => {
    handleFileChange(e);
    const file = e.target.files[0];
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileChange({ target: { files: e.dataTransfer.files } });
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
    setDragOver(false);
  };

  return (
    <div
      className={`uploader-card ${dragOver ? "drag-over" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={onDrop}
    >
      <h2>Déposez une image d’embryon ici</h2>
      <FontAwesomeIcon icon={faUpload} size="3x" style={{ margin: "1rem 0" }} />
      <input type="file" accept="image/*" onChange={onFileChange} />
      {previewUrl && (
        <div style={{ margin: "1rem 0" }}>
          <img
            src={previewUrl}
            alt="Aperçu"
            style={{ maxWidth: "300px", maxHeight: "300px", borderRadius: "8px" }}
          />
        </div>
      )}
      <button onClick={handleSubmit} disabled={loading} className="app-button">
        {loading ? "Prédiction en cours..." : "Lancer la prédiction"}
        <FontAwesomeIcon icon={faArrowRight} style={{ marginLeft: "8px" }} />
      </button>
    </div>
  );
};

export default ImageUploader;