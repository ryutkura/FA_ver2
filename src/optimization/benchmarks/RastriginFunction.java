package optimization.benchmarks;

import java.util.Arrays;

public class RastriginFunction extends BenchmarkFunction {
    
    public RastriginFunction(int dimension, double[] shift) {
        super(dimension, createBounds(-100, dimension), createBounds(100, dimension), shift);
    }
    
    private static double[] createBounds(double value, int dimension) {
        double[] bounds = new double[dimension];
        Arrays.fill(bounds, value);
        return bounds;
    }
    
    @Override
    public double evaluate(double[] position) {
        double sum = 0;
        
        for (int i = 0; i < dimension; i++) {
            double xi = position[i] - shift[i];
            sum += xi * xi - 10 * Math.cos(2 * Math.PI * xi) + 10;
        }
        
        return sum;
    }
}