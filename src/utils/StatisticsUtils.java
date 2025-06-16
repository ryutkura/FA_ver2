package utils;

import java.util.Arrays;

public class StatisticsUtils {
    
    /**
     * 配列の平均値を計算
     */
    public static double calculateMean(double[] values) {
        return Arrays.stream(values).average().orElse(Double.NaN);
    }
    
    /**
     * 配列の標準偏差を計算
     */
    public static double calculateStdDev(double[] values) {
        double mean = calculateMean(values);
        double temp = 0;
        for (double a : values) {
            temp += (a - mean) * (a - mean);
        }
        return Math.sqrt(temp / values.length);
    }
    
    /**
     * 配列の最小値を取得
     */
    public static double getMin(double[] values) {
        return Arrays.stream(values).min().orElse(Double.NaN);
    }
    
    /**
     * 配列の最大値を取得
     */
    public static double getMax(double[] values) {
        return Arrays.stream(values).max().orElse(Double.NaN);
    }
    
    /**
     * 配列の中央値を計算
     */
    public static double calculateMedian(double[] values) {
        double[] sortedValues = values.clone();
        Arrays.sort(sortedValues);
        int middle = sortedValues.length / 2;
        if (sortedValues.length % 2 == 0) {
            return (sortedValues[middle-1] + sortedValues[middle]) / 2.0;
        } else {
            return sortedValues[middle];
        }
    }
}
