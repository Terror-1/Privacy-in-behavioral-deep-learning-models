import random
import numpy as np
def dp_analysis_sum_test(dataset_column ,step_size,iterations):
     "compute the diff private sum along the column"
     """
     args:
        dataset_column: the column of the dataset
        step_size: the step size of the eps from 0 to 1
        iterations: the number of iterations per eps
     return:
        avg_dp_result: the average of the diff private sum
        avg_error: the average of the percentage error
        avg_eps: the average of the eps

    example:
        avg_dp_result, avg_error, avg_eps = dp_analysis_sum_test(db['X'], 0.1, 100)   
     """
     maximum = np.max(dataset_column)*1.01
     minimum = np.min(dataset_column)*0.99
     sensitivity = maximum - minimum
     real_sum = np.sum(dataset_column)
     diff_private_sum = []
     percentage_error = []
     used_eps = []
     avg_error = []
     avg_dp_result = []
     noise_arr=[]
     avg_noios = []
     for eps in np.arange(0.01, 1, step_size):
          used_eps.append(eps)
          for i in range(iterations):
            scale = sensitivity/eps
            noise = np.random.laplace(0, scale, 1)
            dp_sum = real_sum + noise
            diff_private_sum.append(dp_sum)
            percentage_error.append(abs((dp_sum - real_sum)/real_sum)*100)
            noise_arr.append(noise)
          avg_noios.append(np.mean(noise_arr))
          avg_dp_result.append(np.mean(diff_private_sum))
          avg_error.append(np.mean(percentage_error))
          percentage_error = []
          diff_private_sum = []
          noise_arr=[]
     return avg_dp_result, avg_error, used_eps,avg_noios
          

          
def dp_analysis_mean_test(dataset_column ,step_size,iterations):
     "compute the diff private sum along the column"
     """
     args:
        dataset_column: the column of the dataset
        step_size: the step size of the eps from 0 to 1
        iterations: the number of iterations per eps
     return:
        avg_dp_result: the average of the diff private sum
        avg_error: the average of the percentage error
        avg_eps: the average of the eps

    example:
        avg_dp_result, avg_error, avg_eps = dp_analysis_sum_test(db['X'], 0.1, 100)   
     """
     maximum = np.max(dataset_column)*1.01
     minimum = np.min(dataset_column)*0.99
     sensitivity = maximum - minimum/len(dataset_column)
     real = np.mean(dataset_column)
     diff_private_sum = []
     percentage_error = []
     noise_arr=[]
     used_eps = []
     avg_error = []
     avg_dp_result = []
     avg_noios = []
     for eps in np.arange(0.01, 1, step_size):
          used_eps.append(eps)
          for i in range(iterations):
            scale = sensitivity/eps
            noise = np.random.laplace(0, scale, 1)
            dp_sum = real + noise
            diff_private_sum.append(dp_sum)
            percentage_error.append(abs((dp_sum - real)/real)*100)
            noise_arr.append(noise)
          avg_dp_result.append(np.mean(diff_private_sum))
          avg_error.append(np.mean(percentage_error))
          avg_noios.append(np.mean(noise_arr))
          noise_arr = []
          percentage_error = []
          diff_private_sum = []
     return avg_dp_result, avg_error, used_eps,avg_noios



