data_path = "app/data"

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 score",
    "MCC",
    "Cohen’s kappa",
    "AUC",
]


class ModelColumns:
    title = "title"
    description = "description"
    budget = "budget"
    hourlyRangeMin = "hourlyRangeMin"
    isHourlyPayment = "isHourlyPayment"
    country = "country"
    category = "category"
    workload = "workload"
    duration = "duration"
    clientTotalCharge = "clientTotalCharge"
    clientTotalJobsPosted = "clientTotalJobsPosted"
    clientFeedbackScore = "clientFeedbackScore"
    clientPastHires = "clientPastHires"
    isPaymentMethodVerified = "isPaymentMethodVerified"
    skills = "skills"
    target = "target"
