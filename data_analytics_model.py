import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def analyze_screen_social_time(df):
    if 'Screen_Time_Hours' not in df.columns or 'Social_Media_Hours' not in df.columns:
        print("Error: 'Screen_Time_Hours' or 'Social_Media_Hours' column not found.")
        return

    inconsistent_entries = df[df['Screen_Time_Hours'] < df['Social_Media_Hours']]

    if not inconsistent_entries.empty:
        print(f"Found {len(inconsistent_entries)} entries where Screen_Time_Hours < Social_Media_Hours:")
        print(inconsistent_entries)

        print("\nPossible reasons for these inconsistencies (as discussed previously):")
        print("- Definition differences (e.g., social media as a subset, background usage)")
        print("- Data collection issues (e.g., errors, glitches, different measurement methods)")
        print("- User behavior (e.g., multitasking, using other devices)")
        print("- Dataset construction (e.g., combining data from different sources)")

        print("\nDescriptive statistics of the inconsistent entries:")
        print(inconsistent_entries[['Screen_Time_Hours', 'Social_Media_Hours']].describe())

    else:
        print("No entries found where Screen_Time_Hours < Social_Media_Hours.")


def calculate_data_accuracy(df):
    if 'Screen_Time_Hours' not in df.columns or 'Social_Media_Hours' not in df.columns:
        print("Error: 'Screen_Time_Hours' or 'Social_Media_Hours' column not found.")
        return None

    consistent_entries = df[df['Screen_Time_Hours'] >= df['Social_Media_Hours']]
    total_entries = len(df)

    if total_entries == 0:
        return 0
    accuracy = (len(consistent_entries) / total_entries) * 100
    return accuracy


def load_and_analyze_data(filepath_or_url):
    try:
        df = pd.read_csv("mental_health_analysis.csv")
        print("\nChecking for missing values:")
        print(df.isnull().sum())
        df_cleaned = df.dropna()
        print(f"\nRemoved {len(df) - len(df_cleaned)} rows with missing values.")
        df = df_cleaned
        analyze_screen_social_time(df)
        accuracy = calculate_data_accuracy(df)
        if accuracy is not None:
            print(f"\nData Accuracy: {accuracy:.2f}% (based on Screen_Time_Hours >= Social_Media_Hours)")
        if 'Gender' in df.columns:
            le = LabelEncoder()
            df['Gender_Encoded'] = le.fit_transform(df['Gender'])
            print("\nGender column encoded.")
            print(df[['Gender', 'Gender_Encoded']].head())
        else:
            print("\nGender column not found. Skipping encoding.")
        if 'Survey_Stress_Score' in df.columns and 'Wearable_Stress_Score' in df.columns:
            df['Combined_Stress_Score'] = (df['Survey_Stress_Score'] + df[
                'Wearable_Stress_Score']) / 2.0  # calculate the average
            print("\nCombined Stress Score calculated and added to DataFrame.")
            print(df[['Survey_Stress_Score', 'Wearable_Stress_Score',
                      'Combined_Stress_Score']].head())  # print the first 5 rows to check the new column
            # Random Forest
            if 'Combined_Stress_Score' in df.columns and 'Screen_Time_Hours' in df.columns:
                X = df[['Combined_Stress_Score']]  # Features
                y = df['Screen_Time_Hours']  # Target

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(f"\nRandom Forest Regression Results (Screen_Time_Hours vs. Combined_Stress_Score):")
                print(f"Mean Squared Error: {mse:.2f}")
                print(f"R-squared: {r2:.2f}")
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                print("\nPredicted vs. Actual Values:")
                print(results_df.head(10))

            else:
                print(
                    "\nCannot perform Random Forest regression. Missing 'Combined_Stress_Score' or 'Screen_Time_Hours' columns.")
            df.to_csv("mental_health_analysis.csv", index=False)
            print("\nDataFrame saved to CSV with Combined_Stress_Score.")
        else:
            print(
                "\nCannot create Combined_Stress_Score. Missing 'Survey_Stress_Score' or 'Wearable_Stress_Score' columns.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath_or_url}")
    except Exception as e:
        print(f"An error occurred: {e}")


load_and_analyze_data("mental_health_analysis.csv")
