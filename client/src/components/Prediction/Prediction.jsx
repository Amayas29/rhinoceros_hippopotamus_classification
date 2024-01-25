import React from "react";

import "./Prediction.scss";

const Prediction = ({ prediction, precision }) => {
  return (
    <section className="prediction">
      <div>
        {"The image provided is classified as "} <span>{prediction}</span>
        {" with a precision of  "} <span>{precision}%</span>
      </div>
    </section>
  );
};

export default Prediction;
