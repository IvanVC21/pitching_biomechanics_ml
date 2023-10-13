# pitching_biomechanics_ml

## Introduction
The [OpenBiomechanics Project](https://www.openbiomechanics.org/) is an amazing initiative by [Driveline Baseball](https://www.drivelinebaseball.com/) Research and Development, as they want to democratize access to elite-level motion capture data. In this case, they offer a sample of a larger dataset they have.
Specifically, in this project I framed it as a regression project to predict pitching velocity using more than 70 biomechanical markers. If your passion is baseball or biomechanics or Machine Learning, I highly encourage you to work with this data.
My best results on Test set were 1.15 RMSE and 0.93 R2 score.

## Instructions to run my code
1-. Download the repo to your local machine.

2-. Go to the repo folder in command line and [activate the virtual environment](https://www.infoworld.com/article/3239675/virtualenv-and-venv-python-virtual-environments-explained.html)

3-. Install the required libraries using `pip install -r requirements.txt` and go back to main project folder using `cd..`

4-. Run the script in main folder using `python main.py`

5-. Once the script finish to executed you will get the RMSE and R2 scores on Test set printed, a pickle model, a feature importance plot image saved and a predictions csv that compares predicted speed vs actual speed and its respective error.

## Technologies used
Project was created using:
  * Python
  * Pandas
  * Numpy
  * Matplotlib
  * Optuna
  * Scikit-learn
  * XGBoost

If you have any question, suggestion or comment you can shoot me an email anytime at jesusivan.vc21@gmail.com or you can contact me via [LinkedIn.](https://www.linkedin.com/in/ivanverdugo-analytics/)



Wasserberger KW, Brady AC, Besky DM, Jones BR, Boddy KJ. The OpenBiomechanics Project: The open source initiative for anonymized, elite-level athletic motion capture data. (2022).
  
