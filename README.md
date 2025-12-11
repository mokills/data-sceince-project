# data-sceince-project
NYC Renting vs Owning Project

This project looks at how many people rent vs own in New York City and what factors matter the most. I used two datasets called all units and occupied and merged them using the column CONTROL because both files had the same ID numbers like 12130002. The all-units dataset told me if the home was rented or owned (TENURE = 1 renter, 2 owner). The occupied dataset had the BORO and HHINC_REC1 income columns. I used a codebook to understand what every column meant since it explains all the numbers in the dataset.

What I Did

Merged both datasets using CONTROL

Used the columns TENURE, BORO (1–5), and income (HHINC_REC1)

Made a bar chart showing renters vs owners in each borough

Made a histogram to show income differences between renters and owners

Made a choropleth map using Folium to show which boroughs have more renters

Ran two machine learning models: Decision Tree and Random Forest

Used Feature Importance, Confusion Matrix, and ROC Curve to evaluate the model

Models

The Decision Tree splits the data based on income and borough to try to separate renters and owners. Random Forest builds many trees and averages them so the results are better and not overfitting. They both show which features are most important, and income was slightly more important than borough.

How to Run This Project

Install the packages:

pip install pandas matplotlib seaborn scikit-learn folium


Run the main file:

python hhhhhuu.py


This loads the data, merges it, creates the graphs, makes the choropleth map, and runs the models.

Output

The NYC map is saved as an HTML file in the outputs folder. The graphs and model results show the patterns of renting vs owning across NYC.

What I Learned

NYC has way more renters than owners, especially in Manhattan and Brooklyn. Only extremely high-income people tend to own, and even then the number is small. Borough also matters, with Staten Island being the only place where owning is more common. Most New Yorkers are renters no matter what.

Author

Moshe Levinson
Hunter College
