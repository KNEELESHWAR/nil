# Loan Approval Prediction Web Application

This is a Flask-based web application that predicts loan approval status based on user inputs such as gender, marital status, applicant income, loan amount, credit history, and CIBIL score. The model used is a Random Forest Classifier trained on a loan prediction dataset.

## Features

- Simple and user-friendly web interface using Flask and Bootstrap.
- Takes multiple user inputs for loan prediction.
- Provides real-time loan approval prediction based on machine learning.

## Dataset

The dataset used for training the model is provided in `train_u6lujuX_CVtuZ9i (1).csv`. The data is preprocessed to handle missing values and encode categorical features.

## Project Structure

```bash
project_folder/
    ├── app.py
    ├── train_u6lujuX_CVtuZ9i (1).csv
    └── templates/
        └── index.html
```

- `app.py`: Main Flask application file that contains the logic for data preprocessing, model training, and prediction.
- `train_u6lujuX_CVtuZ9i (1).csv`: Dataset used for training the model.
- `requirements.txt`: Python dependencies for the project.
- `templates/index.html`: HTML file for the user interface.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Requirements

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-Learn
- Jinja2 (Flask dependency for rendering templates)

## Model

The model used for loan approval prediction is a Random Forest Classifier from the `sklearn` library. It is trained with the following features:

- Gender
- Married
- Applicant Income
- Loan Amount
- Credit History

The prediction is influenced by an additional input, **CIBIL Score**, to refine the model’s prediction based on financial credibility.

## Usage

1. Enter the required details such as gender, marital status, applicant income, loan amount, credit history, and CIBIL score.
2. Click on the **Predict Loan Status** button.
3. The predicted result (Yes/No) will be displayed on the screen.

## Screenshots

### Home Page
![Home Page](screenshots/home_page.png)

### Prediction Result
![Prediction Result](screenshots/prediction_result.png)

## Contributing

Contributions are welcome! Feel free to raise an issue or submit a pull request for any improvements or feature suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is sourced from public loan prediction datasets available online.
- Thanks to the contributors of the libraries used in this project, such as Flask, Pandas, NumPy, and Scikit-Learn.

## Contact

If you have any questions, feel free to contact me at `11030.neeleshwar@gmail.com`.
```

### Explanation:

1. **Features**: Listed the key features of the web app for clarity.
2. **Dataset**: Added a brief note about the dataset used.
3. **Project Structure**: Provided a simple tree diagram of the project structure to help users navigate the codebase.
4. **Installation**: Detailed steps to clone the repo, install dependencies, and run the application.
5. **Requirements**: Listed all necessary Python packages for running the project.
6. **Model**: Explained the model and its features briefly, including how the CIBIL score is used.
7. **Usage**: Steps on how to use the web interface for predicting loan approval.
8. **Screenshots**: Mentioned screenshots of the application for better visual understanding.
9. **Contributing** and **License**: Standard sections for open-source projects. Replace the license section with the appropriate license if needed.
10. **Contact**: Added a placeholder for contact details. Update it with your information. 

Feel free to modify any of these sections according to your project specifics.
