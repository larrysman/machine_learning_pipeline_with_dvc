## `End-To-End Machine Learning Pipeline with Data Version Control (DVC)`
#### `Project Overview`
This project leverages on the understanding of developing an end-to-end machine learning models for a typical classification problems but more importantly on the use of `data version control` for producing machine learning pipeline.

This project demonstrates a `reproducible, modular, and version-controlled machine learning pipeline` leveraging on data version control `dvc`, available here:[DVC](https://dvc.org/).

This pipeline uses DVC to manage data, models, and experiment artifacts, while Git handles code and workflow definitions. This ensures the workflow is collaborative, auditable, and easy to reproduce at any time, allowing for collaboration and improvement.

`Short Notes on DVC`:
*Versioning large data files and directories for data science is powerful, but often not enough. Data needs to be filtered, cleaned, and transformed before training ML models - for that purpose DVC introduces a build system to define, execute and track data pipelines — a series of data processing stages, that produce a final result. DVC is a `Makefile` system for machine learning projects!*
*DVC pipelines are versioned using Git, and allow you to better organize projects and reproduce complete workflows and results at will. You could capture a simple ETL workflow, organize your project, or build a complex `DAG (Directed Acyclic Graph)` pipeline.*

#### `Data Available`
The purpose of this project is just to demonstrates an end-to-end machine learning pipeline with dvc and hence a very simple dataset on the room occupancy for a given entity and much information are not explicitly defined here.

The sample snapshot of the dataset is attached below:

<img width="460" height="116" alt="image" src="https://github.com/user-attachments/assets/f49251e1-13da-49fb-8920-cbba620353bd" />

#### `Project Structure and Directory`
For the purpose of a structured project workflows, I have adapted the following as the structure for this project:
- **`src/`** &mdash; *Contains modular, single-responsibility Python scripts for each pipeline stage.*
- **`data/`** &mdash; *Contains raw datasets and all other datasets that produced from each stage of the pipeline whose outputs are data and they are typically DVC-tracked but not included in Git history.*
- **`models/`** &mdash; *Stores all preprocessed and trained model artifacts and are tracked by dvc (DVC-tracked).*
- **`dvc.yaml`** &mdash; *Defines the pipeline stages, dependencies, inputs, and outputs.*
- **`.dvc/`** &mdash; *Internal DVC configuration.*
- **`requirements.txt`** &mdash; *For Python dependencies and need to run and install before reproducing this work.*
- **`.gitignore` / `.dvcignore`** &mdash; *Lists files/directories ignored by Git or DVC.*

#### `Machine Learning Pipeline Stages`
For this project the machine learning **stages** include:
1. **`Data Collection`**: *Ingest raw data from source.*
2. **`Data Preparation`**: *Clean, transform, and ensuring the dataset contains no missing or duplicates values.*
3. **`Feature Engineering`**: *Engineer important and relevant features that may improve the performance of the models and aid analysis.*
4. **`Data Preprocessing`**: *Perform Numerical features tranformation and categorical features encoding using the appropriate scalers and encoders.*
   
   - **`All data structure for stages with data as outputs`**

     <img width="505" height="274" alt="image" src="https://github.com/user-attachments/assets/6f0bf5ae-97ec-4b84-b5b1-fea37bb13b6d" />

5. **`Model Building`**: *Train machine learning models pn the training dataset.*

   - **`Models and Artifacts`**

     <img width="576" height="104" alt="image" src="https://github.com/user-attachments/assets/41b74907-1b78-4b36-8c0c-987fd6e100e0" />

6. **`Model Evaluation`**: *Evaluate model performance and generate metrics.*

   - **`Metrics`**
     
     <img width="510" height="52" alt="image" src="https://github.com/user-attachments/assets/dfad6ebc-ac45-461f-8eca-bbed1604b4df" />

7. **`Deployment/Finalization`**: *Export or register final models or predictions.*

    - **`Visualizing the Stage's Structure`, run:** `dvc dag`

     <img width="730" height="578" alt="image" src="https://github.com/user-attachments/assets/ac1e33fe-387e-4ff3-aa83-e10c75e5a859" />

    - **`Visualizing Outputs of stage`, run:** `dvc dag --outs`

      <img width="794" height="579" alt="image" src="https://github.com/user-attachments/assets/c23e5f6f-3ccf-4a78-9585-032b3ea9014a" />

    - **`Connection FlowChart`, run:** `dvc dag --mermaid`

      <img width="759" height="225" alt="image" src="https://github.com/user-attachments/assets/6a1c31c7-f9d2-4cca-a37b-3fddd665366d" />


`Please note that`: *Each stage is executed, tracked, and versioned by DVC, ensuring all outputs are reproducible and auditable.*

#### `Project Setup Instructions`
- **`Set up the Python Virtual Environment`**: `python -m venv name_of_your_virtual_environment`
- **`Activate the Python Virtual Environment`**:
  - `source name_of_your_virtual_environment/bin/activate` - *for MacOS*
  - `name_of_your_virtual_environment\Scripts\activate` - *for Windows*
- **`Clone the Repository`**: `https://github.com/larrysman/machine_learning_pipeline_with_dvc.git`
- **`Install Project Dependencies`**: `pip install -r requirements.txt`
- **`Configure DVC remote Storage`**: *DVC requires external storage for large files and artifacts. The external storage can be AWS S3 bucket, GDrive, Azure etc for collaboration purposes but for this project just create your own external storage locally:* `mkdir dvcstore`
  - `run`: `dvc remote add -d myremote ./dvcstore` or `dvc remote add -d myremote /path/to/your/dvcstore`.
- **`Pull data and models tracked by DVC`**: `dvc pull`
- **`Running and Reproducing the Pipeline`**: `dvc repro`

`DVC will automatically run each stage sequentially, respecting dependencies and all intermediate and final outputs are tracked; previous results are cached for faster runs.`

#### `Model Evaluation`
To view the evaluation metrics for this project run: `dvc metrics show model_evaluation/classification_metrics/model_metrics.json`

<img width="745" height="36" alt="image" src="https://github.com/user-attachments/assets/99cd7e66-c5ff-4ab5-a9dc-693cb673ec55" />

  - **`Confusion Matrix`**:

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/ff45feb9-9516-484f-afa8-d5290dc41f81" />

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/49a47256-9b15-436b-9bc4-6a07c1278133" />

  - **`Compared Result with Other Versions or Branches`**: `dvc metrics diff`

#### `Contributions and Collaboration` 👩🏻‍🤝‍👨🏽
For any contributions or collaboartion, ensure you follow the procedures below:

1. You **change data or code** (for instance, in `src/`) and to **track new or updated data files do**: `dvc add data/<new-data-file>`
2. You will need to **reproduce the pipeline** to re-run all stages as needed and do: `dvc repro`
3. You will need to **commit code and DVC meta-files, hence do the following**:
   - `git add .`
   - `git commit -m "Describe your changes explicitly"`
   - `git push`
4. You will need to **push data and models to DVC remote do**: `dvc push`


`Please note:` *Remember, contributing to open source projects is more than just a code. You can also contribute by reporting bugs, suggesting new features, improving documentation, and more. Thank you for considering contributing to this project!!!🙌🙌🙌*

🔚

#### `Author`

`Email`: `Larrysman2004@yahoo.com`

`Name`: `Olanrewaju Adegoke`

©️`O L A L Y T I C S`












