"use client"; // Required for client-side interactions

import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPredictions(response.data.predictions);
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Face Classification</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      <button onClick={handleUpload} style={{ margin: "10px", padding: "10px" }}>
        Predict
      </button>

      {predictions && (
        <div>
          <h2>Predictions:</h2>
          {predictions &&
            Object.entries(predictions).map(([name, confidence]) => (
              <p key={name}>
                <strong>{name}:</strong> {parseFloat(confidence).toFixed(2)}%
              </p>
            ))}
        </div>
      )}
    </div>
  );
}
