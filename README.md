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

## Data

| Variable    | Variable Type | Description                                            |
| ----------- | ------------- | -------------------------------------------------------|
| country     | chr           | Geographical location                                  |
| year        | num           | Year of observation                                    | 
| co2         | num           | Annual total emissions of carbon dioxide (million tonnes, excluding land-use change) |



Data file can be downloaded through this link: https://github.com/owid/co2-data#%EF%B8%8F-download-our-complete-co2-and-greenhouse-gas-emissions-dataset--csv--xlsx--json


## Figures

### Project 2 Figures Table of Contents
| Figure Name      | Description |
| ----------- | ----------- |
| Image_Set_1.png | First sample run of predictive model, contains 9 images and its predictions, all the predictions were correct|
| Image_Set_2.png | Second sample run of predictive model, contains 9 images and its predictions, all the predictions were correct|
| Image_Set_3.png | Third sample run of predictive model, contains 9 images and its predictions, all the preditions were correct| 
| Training_Validation_Graph.png | Learning curves for the predictive model, created by plotting training and validation errors (losses) and accuracies against the number of epochs|
| Model Comparison.png | Learning curves showing the loss and accuracy for our predictive models|

View figures here: https://github.com/dtb9de/DS4002P2/tree/main/Project2_Images

Model Values
Model: "sequential"


## References
[1] Hannah Ritchie, Max Roser and Pablo Rosado (2020) - "CO₂ and Greenhouse Gas Emissions". Published online at OurWorldInData.org. Retrieved from: https://ourworldindata.org/co2-and-greenhouse-gas-emissions 

[2] “The Science of Climate Change,” The Science of Climate Change | The world is warming Wisconsin DNR, https://dnr.wisconsin.gov/climatechange/science (accessed Nov. 8, 2023). 

### Previous Submissions
MI1:  
MI2:  

