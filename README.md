# MarketMaven
Predict Stock Market Prices

1. Clone repository (In case if you are importing the project from Github, else skip):

    $ git clone https://github.com/Arvind-AI-7/MarketMaven.git


2. Setup Environment repository:

    $ python -m venv mmenv

    $ source mmenv/bin/activate               [LINUX/MAC]

    $ .\mmenv\Scripts\activate                 [WINDOWS]

    $ pip install -r src/requirements.txt

3. Run the code:

   $ python .\src\main.py

   Note: If you want to retrain the model for some other stock, or the
         current model has become outdated and giving wrong predictions
         for the AAPL stock, you can retrain the model by some modifications 
         in the "main.py" file, by setting the "tt_switch" variable to 0. 
         (0 for training the model and 1 for testing the model). Also, you 
         can set the "years" variable to extract the data for required number of 
         years, and "split" variable to set the train-test split percentage.

4. Some of the results will be printed, while some will be stored in
   (.\src\MarketMaven\screenshots\) directory.

