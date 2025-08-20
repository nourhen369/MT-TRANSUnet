import React, { useState } from 'react';
import { Link } from 'react-scroll';
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowDown } from "@fortawesome/free-solid-svg-icons";
import './App.css';
import ImageUploader from './components/ImageUploader';
import PredictionDisplay from './components/PredictionDisplay';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predMask, setpredMask] = useState(null);
  const [classification, setClassification] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setpredMask(null);
    setClassification(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) return alert("Veuillez sélectionner une image.");

    setLoading(true);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setpredMask(data.masks);
      setClassification(data.classification);
    } catch (error) {
      alert("Erreur lors de la prédiction");
      console.error(error);
    }

    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-card">
          <p>
            Projet basé sur le papier <a href="https://arxiv.org/abs/2102.04306" target="_blank" rel="noreferrer">
              TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
            </a>
          </p>
          <p>
            Cet outil sert à segmenter les images médicales d’<b>embryons</b> et à prédire leur aptitude à l’incubation lors de la <b>FIV</b>, basé sur le dataset référencé dans l'article d'IEEE <a href="https://ieeexplore.ieee.org/document/8059868" target="_blank" rel="noreferrer">Automatic Identification of Human Blastocyst Components via Texture</a>.
          </p>
          <button className="app-button">
            <Link
              to="upload-section"
              smooth={true}
              duration={1000}
              offset={-50}
              className="app-button"
            >
              Faire une Prediction <FontAwesomeIcon icon={faArrowDown} style={{ marginLeft: '8px' }} />
            </Link>
          </button>
        </div>    
        <div className="contact-bar">
          Contact : <a href="mailto:nourhan.khechine@insat.ucar.tn">nourhan.khechine@insat.ucar.tn</a>
        </div>
      </header>

      <footer id="upload-section" className="App-footer">
        <div className="footer-columns">
          <div className="footer-column">
            <ImageUploader
              handleFileChange={handleFileChange}
              handleSubmit={handleSubmit}
              loading={loading}
            />
          </div>
          <div className="footer-column">
            <PredictionDisplay predMask={predMask} classification={classification} />
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;