import pandas as pd
import numpy as np

# Years elections took place in the UK. Used throughout
electionYears = [2015, 2017, 2019]

# Data for House Prices
# Extract only the columns we are interested in
columnsHousePrices = ["ONSConstID", "ConstituencyName", "CountryID", "DateOfDataset", "HouseConstMedianPrice"]
housePriceData = pd.read_csv('C:/Programming/cs350/data/housePrices.csv', skipinitialspace=True, usecols=columnsHousePrices, parse_dates=[4],infer_datetime_format=True)
# Convert all dates to years only
housePriceData["DateOfDataset"] = pd.DatetimeIndex(housePriceData["DateOfDataset"]).year
# Extract only election years
housePriceData = housePriceData[housePriceData["DateOfDataset"].isin(electionYears)]
# Remove empty values
housePriceData = housePriceData[housePriceData["HouseConstMedianPrice"] != "-"]
# Split data by election year
housePriceData2015 = housePriceData[housePriceData["DateOfDataset"] == 2015]
housePriceData2017 = housePriceData[housePriceData["DateOfDataset"] == 2017]
housePriceData2019 = housePriceData[housePriceData["DateOfDataset"] == 2019]
# Convert values to floats
housePriceData2015["HouseConstMedianPrice"] = housePriceData2015["HouseConstMedianPrice"].astype(float)
housePriceData2017["HouseConstMedianPrice"] = housePriceData2017["HouseConstMedianPrice"].astype(float)
housePriceData2019["HouseConstMedianPrice"] = housePriceData2019["HouseConstMedianPrice"].astype(float)
# Average values out over a year
housePriceData2015 = housePriceData2015.groupby(["ONSConstID"])["HouseConstMedianPrice"].mean()
housePriceData2017 = housePriceData2017.groupby(["ONSConstID"])["HouseConstMedianPrice"].mean()
housePriceData2019 = housePriceData2019.groupby(["ONSConstID"])["HouseConstMedianPrice"].mean()
# Convet back to dataframe
housePriceData2015 = pd.DataFrame(housePriceData2015)
housePriceData2015.reset_index(inplace=True)
housePriceData2017 = pd.DataFrame(housePriceData2017)
housePriceData2017.reset_index(inplace=True)
housePriceData2019 = pd.DataFrame(housePriceData2019)
housePriceData2019.reset_index(inplace=True)

# Data for House Tenure
# Extract only the columns we are interested in
columnsHousingTenure = ["ONSConstID", "ConstituencyName", "CountryID", "CON%Own"]
housingTenureData = pd.read_csv('C:/Programming/cs350/data/housingTenure.csv', skipinitialspace=True, usecols=columnsHousingTenure)
# Data only from 2011 census so only one data set needed

# Data for Education
# Extract only the columns we are interested in
columnsEducation = ["Constituency ID", "Constituency Name", "Year", "Cons. A*-C %"]
educationData = pd.read_csv('C:/Programming/cs350/data/education.csv', skipinitialspace=True, usecols=columnsEducation, parse_dates=[3], infer_datetime_format=True)
# Extract only election years
educationData = educationData[educationData["Year"].isin(electionYears)]
# Remove the NaN values
educationData = educationData.dropna()
educationData = educationData.rename(columns={"Constituency ID": "ONSConstID", "Constituency Name": "ConstituencyName"})
# Split data by election year
educationData2015 = educationData[educationData["Year"] == 2015]
educationData2017 = educationData[educationData["Year"] == 2017]
educationData2019 = educationData[educationData["Year"] == 2019]
educationData2015 = educationData2015.drop(columns=["Year"])
educationData2017 = educationData2017.drop(columns=["Year"])
educationData2019 = educationData2019.drop(columns=["Year"])
# Portsmouth North was missing data in 2019
# Converting percentages to floats
educationData2015["Cons. A*-C %"] = educationData2015["Cons. A*-C %"].str.rstrip('%').astype('float') / 100.0
educationData2017["Cons. A*-C %"] = educationData2017["Cons. A*-C %"].str.rstrip('%').astype('float') / 100.0
educationData2019["Cons. A*-C %"] = educationData2019["Cons. A*-C %"].str.rstrip('%').astype('float') / 100.0


# Data for age
# Extract only the data we are interested in
columnsAge = ["ONSConstID", "ConstituencyName", "CountryID", "Date", "Const%", "Age group"]
ageData = pd.read_csv('C:/Programming/cs350/data/age.csv', skipinitialspace=True, usecols=columnsAge, parse_dates=[4], infer_datetime_format=True)
# Split dataset based on years required
ageData["Date"] = pd.DatetimeIndex(ageData["Date"]).year
ageData2015 = ageData[ageData["Date"] == 2015]
ageData2017 = ageData[ageData["Date"] == 2017]
ageData2019 = ageData[ageData["Date"] == 2019]

# Pivot the data so that each constituency is a single record
ageData2015 = ageData2015.pivot(index='ONSConstID', columns='Age group', values='Const%')\
            .reset_index()
ageData2015.columns.name=None
ageData2017 = ageData2017.pivot(index='ONSConstID', columns='Age group', values='Const%')\
            .reset_index()
ageData2017.columns.name=None
ageData2019 = ageData2019.pivot(index='ONSConstID', columns='Age group', values='Const%')\
            .reset_index()
ageData2019.columns.name=None
# Convert percentages to floats
ageGroups = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
for age in ageGroups:
    ageData2015[age] = ageData2015[age].str.rstrip('%').astype('float') / 100.0
    ageData2017[age] = ageData2017[age].str.rstrip('%').astype('float') / 100.0
    ageData2019[age] = ageData2019[age].str.rstrip('%').astype('float') / 100.0

# Data for country of birth
# Extract only the columns we are interested in
columnsBirthCountry = ["ONSConstID", "ConstituencyName", "CountryID", "Con%Europe:UnitedKingdom:England", "Con%Europe:UnitedKingdom:NorthernIreland", "Con%Europe:UnitedKingdom:Scotland",
                       "Con%Europe:UnitedKingdom:Wales", "Con%Europe:ChannelIslandsandIsleofMan", "Con%Europe:Ireland", "Con%Europe:OtherEurope:EUCountries:Total", "Con%Europe:OtherEurope:RestofEurope:Total",
                       "Con%Africa:NorthAfrica", "Con%Africa:CentralandWesternAfrica:Total", "Con%Africa:SouthandEasternAfrica:Total", "Con%MiddleEastandAsia:MiddleEast:Total",
                       "Con%MiddleEastandAsia:EasternAsia:Total", "Con%MiddleEastandAsia:SouthernAsia:Total", "Con%MiddleEastandAsia:South-EastAsia:Total", "Con%MiddleEastandAsia:CentralAsia",
                       "Con%TheAmericasandtheCaribbean:CentralAmerica", "Con%TheAmericasandtheCaribbean:SouthAmerica", "Con%AntarcticaandOceania:Australia", "Con%AntarcticaandOceania:OtherAntarcticaandOceania",
                       "Con%Other"]
birthCountryData = pd.read_csv('C:/Programming/cs350/data/birthCountry.csv', skipinitialspace=True, usecols=columnsBirthCountry)
# Data only from 2011 census so only one data set needed

# Data for deprivation
# Extract data twice, once for each year
columnsDeprivation2015 = ["ONSConstID", "ConstituencyName", "CountryID", "IMD rank 2015"]
deprivationData2015 = pd.read_csv('C:/Programming/cs350/data/deprivation.csv', skipinitialspace=True, usecols=columnsDeprivation2015)
columnsDeprivation2019 = ["ONSConstID", "ConstituencyName", "CountryID", "IMD rank 2019"]
deprivationData2019 = pd.read_csv('C:/Programming/cs350/data/deprivation.csv', skipinitialspace=True, usecols=columnsDeprivation2019)
# Rename columns to be matching
deprivationData2015.rename(columns={"IMD rank 2015": "DeprivationRank"}, inplace=True)
deprivationData2019.rename(columns={"IMD rank 2019": "DeprivationRank"}, inplace=True)
#Use 2015 data for 2017 election
deprivationData2017 = deprivationData2015

# Data for unemployment benefits
# Extract only the columns we want
columnsUnemployment = ["ONSConstID", "ConstituencyName", "CountryID", "DateOfDataset", "UnempConstRate"]
unemploymentData = pd.read_csv('C:/Programming/cs350/data/unemployment.csv', skipinitialspace=True, usecols=columnsUnemployment, parse_dates=[4], infer_datetime_format=True)
# Convert all dates to years only
unemploymentData["DateOfDataset"] = pd.DatetimeIndex(unemploymentData["DateOfDataset"]).year
# Split data by year
unemploymentData2015 = unemploymentData[unemploymentData["DateOfDataset"] == 2015]
unemploymentData2017 = unemploymentData[unemploymentData["DateOfDataset"] == 2017]
unemploymentData2019 = unemploymentData[unemploymentData["DateOfDataset"] == 2019]
# Convert values to floats
unemploymentData2015["UnempConstRate"] = unemploymentData2015["UnempConstRate"].astype(float)
unemploymentData2017["UnempConstRate"] = unemploymentData2017["UnempConstRate"].astype(float)
unemploymentData2019["UnempConstRate"] = unemploymentData2019["UnempConstRate"].astype(float)
# Average values out over a year
unemploymentData2015 = unemploymentData2015.groupby(["ONSConstID"])["UnempConstRate"].mean()
unemploymentData2017 = unemploymentData2017.groupby(["ONSConstID"])["UnempConstRate"].mean()
unemploymentData2019 = unemploymentData2019.groupby(["ONSConstID"])["UnempConstRate"].mean()
# Convert back to dataframe
unemploymentData2015 = pd.DataFrame(unemploymentData2015)
unemploymentData2015.reset_index(inplace=True)
unemploymentData2017 = pd.DataFrame(unemploymentData2017)
unemploymentData2017.reset_index(inplace=True)
unemploymentData2019 = pd.DataFrame(unemploymentData2019)
unemploymentData2019.reset_index(inplace=True)

# Data for child poverty
# Extract only the columns we want
columnsChildPoverty = ["ONSConstID", "ConstituencyName", "CountryID", "Year", "Constituency relative rate"]
childPovertyData = pd.read_csv('C:/Programming/cs350/data/childPoverty.csv', skipinitialspace=True, usecols=columnsChildPoverty, infer_datetime_format=True)
# Split data based on year
childPovertyData2015 = childPovertyData[childPovertyData["Year"] == "2015/16"]
childPovertyData2017 = childPovertyData[childPovertyData["Year"] == "2017/18"]
childPovertyData2019 = childPovertyData[childPovertyData["Year"] == "2019/20"]
# Dropping the year column
childPovertyData2015 = childPovertyData2015.drop(columns=["Year"])
childPovertyData2017 = childPovertyData2017.drop(columns=["Year"])
childPovertyData2019 = childPovertyData2019.drop(columns=["Year"])

# Combining data sets

englandData2015 = housePriceData2015.merge(housingTenureData, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2015 = englandData2015.merge(educationData2015, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2015 = englandData2015.merge(ageData2015, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2015 = englandData2015.merge(deprivationData2015, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2015 = englandData2015.merge(unemploymentData2015, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2015 = englandData2015.merge(childPovertyData2015, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2015 = englandData2015.merge(birthCountryData, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

englandData2017 = housePriceData2017.merge(housingTenureData, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(educationData2017, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(ageData2017, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(deprivationData2017, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(unemploymentData2017, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(childPovertyData2017, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(birthCountryData, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

englandData2019 = housePriceData2019.merge(housingTenureData, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(educationData2019, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(ageData2019, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(deprivationData2019, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(unemploymentData2019, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(childPovertyData2019, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(birthCountryData, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

# Get election, the labels we will predict
# Extract only the columns we want
columnsElections = ["ons_id", "first_party"]
electionData2015 = pd.read_csv('C:/Programming/cs350/data/election2015.csv', skipinitialspace=True, usecols=columnsElections)
electionData2017 = pd.read_csv('C:/Programming/cs350/data/election2017.csv', skipinitialspace=True, usecols=columnsElections)
electionData2019 = pd.read_csv('C:/Programming/cs350/data/election2019.csv', skipinitialspace=True, usecols=columnsElections)

# Rename Columns to match the data we already have
electionData2015 = electionData2015.rename(columns={"ons_id": "ONSConstID", "first_party": "Result"})
electionData2017 = electionData2017.rename(columns={"ons_id": "ONSConstID", "first_party": "Result"})
electionData2019 = electionData2019.rename(columns={"ons_id": "ONSConstID", "first_party": "Result"})

# Combine the data together
englandData2015 = englandData2015.merge(electionData2015, on="ONSConstID", suffixes=('', '_y'))
englandData2015.drop(englandData2015.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2017 = englandData2017.merge(electionData2017, on="ONSConstID", suffixes=('', '_y'))
englandData2017.drop(englandData2017.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
englandData2019 = englandData2019.merge(electionData2019, on="ONSConstID", suffixes=('', '_y'))
englandData2019.drop(englandData2019.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

# Create combined dataset
englandData2015.insert(39, "Year", 2015)
englandData2017.insert(39, "Year", 2017)
englandData2019.insert(39, "Year", 2019)
englandDataFull = pd.concat([englandData2015, englandData2017, englandData2019])

# Export data
englandDataFull.to_csv(r'C:/Programming/cs350/data/englandDataFull.csv', index=False)


