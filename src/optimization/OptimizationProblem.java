// OptimizationProblem.java
package optimization;

public interface OptimizationProblem {
    double evaluate(double[] position);
    double[] getLowerBounds();
    double[] getUpperBounds();
    int getDimension();
}