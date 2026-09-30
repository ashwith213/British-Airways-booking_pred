# British Airways: Predicting Customer Booking Behaviour

Forage virtual job simulation (Data Science). The goal is to predict whether a
customer will complete a flight booking, and to understand which factors drive it.

> This is a job simulation project, not work done for British Airways.

## Dataset
- 50,000 bookings, 13 features, no missing values
- Target: `booking_complete` (1 = completed). Only about 15% of bookings are completed, so the classes are imbalanced.
- Features include sales channel, trip type, days booked ahead, length of stay, flight day and hour, route, booking country, number of passengers, extras requested (baggage, seat, meals) and flight duration.

## Approach
1. **Exploratory analysis:** completion rate by country, lead time, extras, channel, day and hour.
2. **Preprocessing:** one-hot encoding of categorical columns inside a scikit-learn `Pipeline`, so encoding is fitted separately in each fold.
3. **Model:** Random Forest with `class_weight="balanced"` to handle the class imbalance.
4. **Evaluation:** 5-fold stratified cross-validation.
5. **Interpretation:** permutation importance on a held-out 20% test set.

## Results (5-fold cross-validation)
| Metric | Score |
|---|---|
| ROC AUC | 0.763 (+/- 0.005) |
| Recall | 0.759 |
| Precision | 0.276 |
| F1 | 0.405 |
| Accuracy | 0.666 |

Accuracy is low on purpose: with 85% of bookings not completed, a model that always
predicts "no" would score about 85% accuracy and be useless, so AUC and recall matter more here.
The model catches about 3 in 4 real bookers, but only about 28% of the customers it flags convert.

## Key findings
- **Booking country** is by far the strongest predictor (permutation importance about 0.16 AUC drop).
- **Route** (about 0.04) and **length of stay** (about 0.02) come next.
- Most other variables contribute little on their own.

The feature importance ranking comes from a single 80/20 split, so it is directional,
not exact.

## Limitations and next steps
- Low precision means the scores are best used to rank customers, not as certain predictions.
- Try gradient boosting and tune the decision threshold to balance campaign cost against missed bookers.
- `route` has about 800 categories, so grouping rare routes could reduce noise.

## Bonus: Lounge eligibility model
A lookup table estimating the share of passengers eligible for Tier 1, 2 and 3 lounge access,
grouped by haul type and region, built from a 10,000-flight schedule.

## Files
- `notebook.ipynb`: analysis and model
- `BA_Booking_Prediction_Summary.pdf`: one-slide summary
- `customer_booking.csv`: dataset (remove this if the license does not allow sharing it)

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn