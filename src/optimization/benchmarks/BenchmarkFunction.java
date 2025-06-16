package optimization.benchmarks;

import optimization.OptimizationProblem;
// import java.util.Arrays;

/**
 * ベンチマーク関数の基底クラス
 */
public abstract class BenchmarkFunction implements OptimizationProblem {
    protected int dimension;
    protected double[] lowerBounds;
    protected double[] upperBounds;
    protected double[] shift; // 最適解のシフト値
    
    public BenchmarkFunction(int dimension, double[] lowerBounds, double[] upperBounds, double[] shift) {
        this.dimension = dimension;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.shift = shift;
        
        // シフト値が指定されていない場合は0で初期化
        if (this.shift == null) {
            this.shift = new double[dimension];
        }
    }
    
    @Override
    public double[] getLowerBounds() {
        return lowerBounds;
    }
    
    @Override
    public double[] getUpperBounds() {
        return upperBounds;
    }
    
    @Override
    public int getDimension() {
        return dimension;
    }
}