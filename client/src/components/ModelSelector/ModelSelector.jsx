import React from "react";
import Select from "react-select";
import Checkbox from "@mui/material/Checkbox";

import "./ModelSelector.scss";

const ModelSelector = ({
  modelNames,
  currentModel,
  setSelectedModel,
  withColor,
  setwithColor,
}) => {
  const handleChange = () => {
    setwithColor(!withColor);
  };

  return (
    <section className="modelselector">
      <div className="modelselector-selector">
        Model to use for classification :
        <Select
          options={modelNames}
          defaultValue={currentModel}
          onChange={(selectedOption) => {
            setSelectedModel(selectedOption);
          }}
        />
      </div>

      <div>
        Gray the images :
        <Checkbox
          checked={!withColor}
          onChange={handleChange}
          inputProps={{ "aria-label": "controlled" }}
        />
      </div>
    </section>
  );
};

export default ModelSelector;
