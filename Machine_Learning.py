import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pad
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet


class Machine_Learning():

    def __init__(self):
        self.load_data()
    
    def EDA(self,name_df):

        if name_df == "sea_level":
            df_chosen=self.df_sea_level
        elif name_df == "temperature":
            df_chosen=self.df_temperature
        elif name_df == "CO2":
            df_chosen = self.df_CO2_emission
        elif name_df == "merge":
            df_chosen =  self.df_merged
        
        print(f'EDA on the data {name_df}:')
        print("First 5 records:\n", df_chosen.head(),"\n")
        print("Describe:\n",df_chosen.describe(),"\n")
        print("Shape\n", df_chosen.shape,"\n")
        print("Information\n")
        df_chosen.info() #print directly doesn't return anything
        print("Null values:\n", df_chosen.isnull().sum(),"\n")
        print("\n----------------------------------------------------------------------------------------------------------------------------------\n")
        input("Press enter to continue")
        print("\n\n\n\n\n")

    def load_data(self):
        file_path = "city_temperature.csv"
        self.df_temperature = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sudalairajkumar/daily-temperature-of-major-cities",
            file_path,
        )

        file_path = "co2_conc.csv"

        self.df_CO2_emission = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "arunavsutar/daily-atmosphere-carbon-dioxide-concentration",
        file_path,
        )

        file_path = "sealevel.csv"

        self.df_sea_level = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "kkhandekar/global-sea-level-1993-2021",
        file_path,
        )

    def preprocessing_data_sea_level(self):
        self.df_sea_level = self.df_sea_level.drop(columns = ["TotalWeightedObservations","GMSL_noGIA","StdDevGMSL_noGIA","GMSL_GIA","StdDevGMSL_GIA","SmoothedGSML_GIA","SmoothedGSML_noGIA"])
        self.df_sea_level = self.df_sea_level[(self.df_sea_level['Year'] >= 2013) & (self.df_sea_level['Year'] <= 2020)]
        self.df_sea_level = self.df_sea_level.groupby(['Year']).mean().reset_index()

    def preprocessing_data_temperature(self):

        self.df_temperature = self.df_temperature.drop(columns = ["Region","Country","State","City"])
        self.df_temperature = self.df_temperature[(self.df_temperature['Year'] >= 2013) & (self.df_temperature['Year'] <= 2020)]
        self.df_temperature = self.df_temperature.groupby(['Year','Month','Day'])['AvgTemperature'].mean().reset_index()
        self.df_temperature.rename(columns={'AvgTemperature': 'avg_mondial_temp'}, inplace=True)

    def preprocessing_data_CO2(self):
        self.df_CO2_emission.drop(columns = ["Unnamed: 0","cycle"], inplace=True)
        self.df_CO2_emission = self.df_CO2_emission[(self.df_CO2_emission['year'] >= 2013) & (self.df_CO2_emission['year'] <= 2020)]
        self.df_CO2_emission.rename(columns={'year': 'Year'}, inplace=True)
        self.df_CO2_emission.rename(columns={'month': 'Month'}, inplace=True)
        self.df_CO2_emission.rename(columns={'day': 'Day'}, inplace=True)
        self.df_CO2_emission.rename(columns={'trend': 'concentration_in_CO2'}, inplace=True)

    def correlation_merge_df(self):
        
        corr_matrix = self.df_merged.corr()
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(corr_matrix, annot=True) # fmt="d" specifies the annotations' format as decimal integers (d stands for decimal)
        plt.title("Matrice de corrélation")
        plt.show()

    def merge_DataFrame(self):
        self.df_merged = self.df_temperature.merge(self.df_CO2_emission, on=['Year', 'Month', 'Day'], how='inner').merge(self.df_sea_level, on=['Year'], how='inner')

    def predict_Prophet(self):

        df_training = self.df_merged
        df_training['ds'] = pad.to_datetime(df_training[['Year', 'Month', 'Day']])
        df_training.rename(columns={'avg_mondial_temp': 'y'}, inplace=True) #need to rename it for the model
        df_training = df_training[['ds', 'y']]
        
        model = Prophet()
        model.fit(df_training)

        future = model.make_future_dataframe(periods=365 * 30)
        forecast = model.predict(future)

        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        fig = model.plot(forecast)
        plt.legend([
            'Prévision (yhat)', 
            'Incertitude basse (yhat_lower)', 
            'Incertitude haute (yhat_upper)', 
            'Observations'
        ], loc='upper left')
        plt.title("Prévision de la température sur 30 ans")
        plt.xlabel("Date")
        plt.ylabel("Température")
        plt.show()

def Machine_Learning_main():
    Machine_Learning_class=Machine_Learning()
    Machine_Learning_class.EDA("sea_level")
    Machine_Learning_class.EDA("temperature")
    Machine_Learning_class.EDA("CO2")
    Machine_Learning_class.preprocessing_data_sea_level()
    Machine_Learning_class.preprocessing_data_temperature()
    Machine_Learning_class.preprocessing_data_CO2()
    Machine_Learning_class.EDA("sea_level")
    Machine_Learning_class.EDA("temperature")
    Machine_Learning_class.EDA("CO2")
    Machine_Learning_class.merge_DataFrame()
    Machine_Learning_class.EDA("merge")
    Machine_Learning_class.correlation_merge_df()
    Machine_Learning_class.predict_Prophet()


    







if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Option for the Machine_Learning analysis")
    #parser.add_argument("--bucket_staging", type=str, required=True, help="Name of the staging S3 bucket")
    args = parser.parse_args()

    Machine_Learning_main() #args.bucket_staging