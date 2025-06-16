package optimization.benchmarks;

import java.util.Arrays;

public class AckleyFunction extends BenchmarkFunction {
    
    public AckleyFunction(int dimension, double[] shift) {
        super(dimension, createBounds(-100, dimension), createBounds(100, dimension), shift);
    }
    
    private static double[] createBounds(double value, int dimension) {
        double[] bounds = new double[dimension];
        Arrays.fill(bounds, value);
        return bounds;
    }
    
    @Override
    public double evaluate(double[] position) {
        double sum1 = 0;
        double sum2 = 0;
        
        for (int i = 0; i < dimension; i++) {
            double xi = position[i] - shift[i];
            sum1 += xi * xi;
            sum2 += Math.cos(2 * Math.PI * xi);
        }
        
        return -20 * Math.exp(-0.2 * Math.sqrt(sum1 / dimension)) 
             - Math.exp(sum2 / dimension) 
             + 20 + Math.E;
    }
}