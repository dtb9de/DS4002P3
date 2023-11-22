# Project 3: MI3 Group5
MI3 Data Analysis for Project 3 Group 5 for DS 4002

Team Members:
Minh Nguyen(hvn9qwn): Group Leader,
Sally Sydnor(srs8yy),
David Bergman(dtb9de),


## Repository Contents

This repository contains 2 markdown files: README.md and LICENSE.md, as well as 3 folders: SRC, DATA, and Project2_images. The README.md file contains information about the contents of the repo as well as explanations for the src, data, and figures folders. LICENSE.md contains an MIT license for our work. The SRC folder contains the main code file for our project. More information about how the code works will be provided in the next section of this document. The data folder contains instructions on how to download the data file used for this project. A data dictionary is provided in the data section of this readme. The Project2_images folder will contain all of the graphics generated from this project. A description of each figure is provided in the figures section of the readme. Lastly, all of our references will be displayed in the references section of this readme.

## SRC

### Installation/Building of Code

#### Implement the following steps to install and build the code:
1. Importing packages
```{r}
library(readr)
library(dplyr)
library(tidyr)
library(psych)
```
2. Importing and cleaning data
```{r}
knitr::opts_chunk$set(echo = TRUE)

co2_data <- read_csv("~/UVA/Systems/DS 4002/R/owid-co2-data.csv")
View(co2_data)

#str(co2_data)
#summary(co2_data)

#cleaned_data <- na.omit(co2_data)
#View(cleaned_data)

cleaned_data <- distinct(cleaned_data)
names(cleaned_data)

main_data <- co2_data %>%
  dplyr::select("country", "year", "co2", "cement_co2", "population", "gdp", "energy_per_gdp", "land_use_change_co2", "total_ghg")
cleaned_data <- cleaned_data %>%
  dplyr::select("country", "year", "co2", "cement_co2", "population", "gdp", "energy_per_gdp", "land_use_change_co2", "total_ghg")
# Remove global share and cumulative columns
View(main_data)
View(cleaned_data)
```
### Code Usage

Producing linear regression model:

1. Basic country emission model
```{r}
ggplot(data = cleaned_data, aes(x= year, y = co2))+
  geom_line()+
   facet_wrap(~country)

ggplot(data = main_data, aes(x= year, y = co2))+
  geom_line()

main_data %>%
  filter(country == c("China"))%>%
  ggplot(aes(x= year, y = co2))+
  geom_line()
```
2. Highest GHG model and highest correlated variables
```{r}
cleaned_data %>% 
  dplyr::arrange(desc(total_ghg))

??pairs.panels
#correlation
pairs.panels(cleaned_data[,c("total_ghg", "gdp", "population", "energy_per_gdp")])
```
3. Creating Predictive Model
```{r}
co2_lm <- lm(total_ghg~population+year, data=cleaned_data)
summary(co2_lm)

co2_lm_country = lm(total_ghg~population+year+country, data=cleaned_data)
summary(co2_lm_country)

co2_lm_country2 = lm(total_ghg~country, data=cleaned_data)
summary(co2_lm_country2)
```
Machine Learning Model:


## Data

| Variable    | Variable Type | Description                                            |
| ----------- | ------------- | -------------------------------------------------------|
| country     | chr           | Geographical location                                  |
| year        | num           | Year of observation                                    | 
| co2         | num           | Annual total emissions of carbon dioxide (in million tonnes, excluding land-use change) |
| cement_co2  | num           | Annual emissions from carbon dioxide from cement (in million tonnes)                    |
| population  | num           | Population count by country by year                    |
| gdp         | num           | Gross Domesetic Product in international dollars using 2011 prices to adjust for inflation| 
| energy_per_gdp | num        | Primary energy consumption per unit of gross domestic product measured in kilowatt-hours per international dollar | 
| land_use_change_co2 | num   | Annual emissions of carbon dioxide from land-use change (in million tonnes)           |
| total_ghg   | num           | Total greenhouse gas emissions including land-use change and forestry (in million tonnes) | 



Data file can be downloaded through this link: https://github.com/owid/co2-data#%EF%B8%8F-download-our-complete-co2-and-greenhouse-gas-emissions-dataset--csv--xlsx--json


## Figures

### Project 2 Figures Table of Contents
| Figure Name      | Description |
| ----------- | ----------- |
| co2_versus_year.png | Line graph displaying total carbon dioxide emissions by year (1750 - 2018)|
| co2_versus_year2.png | Line graph displaying total carbon dioxide emissions by year (1750 - 2018 |
| correlation.png | Correlation chart showing variable dependencies among all variables | 
| train_data | Distribution chart of train_data conveying breadth of the intersection of the data set |

View figures here: https://github.com/dtb9de/DS4002P2/tree/main/Project2_Images

Model Values
Best accuracy with k=7 (7 nearest neighbors)
[1] "Accuracy for k = 1 : 0.967611336032389"
[1] "Accuracy for k = 3 : 0.983805668016194"
[1] "Accuracy for k = 5 : 0.983805668016194"
[1] "Accuracy for k = 7 : 0.987854251012146"
[1] "Accuracy for k = 9 : 0.983805668016194"


## References
[1] Hannah Ritchie, Max Roser and Pablo Rosado (2020) - "CO₂ and Greenhouse Gas Emissions". Published online at OurWorldInData.org. Retrieved from: https://ourworldindata.org/co2-and-greenhouse-gas-emissions 

[2] “The Science of Climate Change,” The Science of Climate Change | The world is warming Wisconsin DNR, https://dnr.wisconsin.gov/climatechange/science (accessed Nov. 8, 2023). 

### Previous Submissions
MI1: https://myuva-my.sharepoint.com/:w:/r/personal/srs8yy_virginia_edu/Documents/MI1%20Project%203.docx?d=w15b00e2a0f0b4a9591e18b9f849d58c0&csf=1&web=1&e=XqeiMn
MI2: https://myuva-my.sharepoint.com/:w:/r/personal/srs8yy_virginia_edu/Documents/MI2%20Project%203.docx?d=w59de98c7943647288438eab57bb10af6&csf=1&web=1&e=vQd7f3

