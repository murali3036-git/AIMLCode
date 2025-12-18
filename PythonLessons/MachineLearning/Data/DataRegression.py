
from MachineLearning.Model.RegressionModels import InsuranceData


class InsuranceDataProvider:

    @staticmethod
    def GetLinearInsuranceData():
        return [
            InsuranceData(10, 3000),
            InsuranceData(20, 4000),
            InsuranceData(30, 5000),
            InsuranceData(40, 6000),
            InsuranceData(50, 7000),
            InsuranceData(60, 8000),
            InsuranceData(70, 9000),
            InsuranceData(80, 10000),
            InsuranceData(90, 11000),
            InsuranceData(100, 12000),
        ]
