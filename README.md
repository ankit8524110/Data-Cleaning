# Data-Cleaning
The Goal
To take "messy" raw data and polish it so a Machine Learning model can actually learn from it.

What Was Done (The Process)
1. Data Inspection

Loaded the data and looked for holes (missing values), errors, or weird formats.

2. Fixing Missing Information

Age: Instead of deleting passengers with no recorded age, I filled in the median age (the middle value) to keep the data useful.

Cabin: This column had too much missing data to save, so I deleted it.

Embarked (Port): Filled missing spots with the most common port (mode).

3. Translating for the Computer (Encoding)
Computers understand numbers, not words.

Sex: Converted "Male/Female" into 0 and 1.

Embarked: Used "One-Hot Encoding" to turn port names into separate numerical columns.

4. Scaling

Used StandardScaler to adjust numerical values (like Fare and Age) so they are on a similar scale. This prevents one large number from confusing the AI.

5. Quality Control

Used Boxplots to spot "outliers" (data points that look suspiciously extreme).
