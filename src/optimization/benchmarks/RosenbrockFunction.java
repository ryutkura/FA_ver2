package optimization.benchmarks;

import java.util.Arrays;

public class RosenbrockFunction extends BenchmarkFunction {
    
    public RosenbrockFunction(int dimension, double[] shift) {
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
        for (int i = 0; i < dimension - 1; i++) {
            double xi = position[i] - shift[i];
            double xi1 = position[i + 1] - shift[i + 1];
            sum += 100 * Math.pow(xi1 - xi * xi, 2) + Math.pow(xi - 1, 2);
        }
        return sum;
    }
}