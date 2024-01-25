import React, { useState } from "react";

import { ModelSelector, ImagePicker, Prediction } from "./components";

import axios from "axios";

import "./App.scss";

const modelNames = [
  { value: "vit", label: "Pretrained google ViT" },
  { value: "cnn", label: "CNN from scratch" },
  { value: "resnet", label: "Pretrained ResNet18" },
];

const App = () => {
  const [image, setImage] = useState(null);
  const [modelName, setModelName] = useState(modelNames[0]);
  const [withColor, setwithColor] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [precision, setPrecision] = useState(null);

  const [loading, setLoading] = useState(false);

  return (
    <>
      {loading && (
        <div className="loading">
          Loading...
          <div class="loading-ellipsis">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
          </div>
        </div>
      )}
      <div className="app">
        <ImagePicker
          setImage={(img) => {
            setPrediction(null);
            setPrecision(null);
            setwithColor(true);
            setModelName(modelNames[0]);

            setImage(img);
          }}
        />

        <aside>
          <ModelSelector
            modelNames={modelNames}
            currentModel={modelName}
            setSelectedModel={setModelName}
            withColor={withColor}
            setwithColor={setwithColor}
          />
          {console.log(withColor)}
          <button
            onClick={() => {
              if (!image) return;

              const imageBlob = new Blob([image], { type: image.type });
              let data = new FormData();

              data.append("image", imageBlob);
              data.append("model_name", modelName.value);
              data.append("with_color", withColor);

              setLoading(true);

              return axios
                .post("http://localhost:5000/predict", data, {
                  headers: {
                    "Content-Type": "multipart/form-data",
                  },
                })
                .then((response) => {
                  setPrediction(response.data.prediction);
                  setPrecision(response.data.precision);
                  setLoading(false);
                });
            }}
          >
            Predict
          </button>

          {prediction && (
            <Prediction prediction={prediction} precision={precision} />
          )}
        </aside>
      </div>
    </>
  );
};

export default App;
