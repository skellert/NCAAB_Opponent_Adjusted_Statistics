# NCAAB_Opponent_Adjusted_Statistics
This is the code that I wrote for generating opponent adjusted statistics for the March Machine Learning Mania Kaggle competition.

First I would like to give credit to Keith Goldner at NumberFire and Paul Kislanko who wrote the information in the below link for giving me the majority of methodology I used.

http://football.kislanko.com/normalizing.html

The functions in the adjust file are designed to convert the standard box score statistics for college basketball which are provided by Kaggle for the March Machine Learning Mania Kaggle competition into more useful advanced metrics. The goal is to separate teams who have inflated stats from playing poor opponents from the truly talented teams playing tougher competition. The datasets that Kaggle provides can be directly plugged into the functions and the adjusted stats will be produced.

Opponent adjusted point differential is essentially a pure rank of a team. The opponent adjusted individual statistics are each teams ability to produce or prevent said statistics adjusting for an opponent's ability to prevent or produce them respectively.
