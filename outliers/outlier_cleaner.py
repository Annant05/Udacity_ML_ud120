#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here

    for pred, age, net_wo in zip(predictions, ages, net_worths):
        error = pred - net_wo
        cleaned_data.append((age, net_wo, error))

        cleaned_data.sort(key=lambda k: k[2])
    return cleaned_data[:int(len(cleaned_data) * 0.9)]
