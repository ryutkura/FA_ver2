package pso;

import optimization.OptimizationProblem;
import optimization.benchmarks.*;
import java.util.Arrays;

public class PSOApplication {
    
    public static void main(String[] args) {
        // 実験パラメータ
        int dimension = 20;
        int runs = 30;
        int maxIterations = 10000*dimension;
        
        // シフト値の設定
        double[] shift = new double[dimension];
        Arrays.fill(shift, 0.0); // 関数を(50,50,...,50)にシフト
        
        // ベンチマーク関数の作成
        OptimizationProblem[] problems = {
            new SphereFunction(dimension, shift),
            new RosenbrockFunction(dimension, shift),
            new AckleyFunction(dimension, shift),
            new RastriginFunction(dimension, shift)
        };
        
        String[] problemNames = {
            "Sphere",
            "Rosenbrock",
            "Ackley",
            "Rastrigin"
        };
        
        // 各ベンチマーク関数で実験を実行
        for (int p = 0; p < problems.length; p++) {
            OptimizationProblem problem = problems[p];
            String problemName = problemNames[p];
            
            System.out.println("======================================");
            System.out.println("Function: " + problemName);
            System.out.println("Dimension: " + dimension);
            System.out.println("Shift: " + shift[0]);
            System.out.println("======================================");
            
            double[] bestFitnesses = new double[runs];
            long[] runtimes = new long[runs];
            
            // 複数回の実行
            for (int r = 0; r < runs; r++) {
                System.out.println("Run " + (r + 1) + "/" + runs);
                
                // PSOのインスタンス作成
                PSO pso = new PSO(problem);
                pso.setParameters(
                    30,           // swarmSize
                    maxIterations, // maxIterations
                    0.729,        // inertiaWeight
                    1.49445,      // c1 (cognitive parameter)
                    1.49445       // c2 (social parameter)
                );
                
                // アルゴリズムの実行と時間計測
                long startTime = System.currentTimeMillis();
                double[] solution = pso.optimize();
                long endTime = System.currentTimeMillis();
                
                // 結果の保存
                double fitness = problem.evaluate(solution);
                bestFitnesses[r] = fitness;
                runtimes[r] = endTime - startTime;
                
                System.out.println("Best fitness: " + fitness);
                System.out.println("Runtime: " + runtimes[r] + " ms");
                // System.out.println("Best position: " + Arrays.toString(solution));
            }
            
            // 統計の計算と表示
            double meanFitness = Arrays.stream(bestFitnesses).average().orElse(Double.NaN);
            double stdFitness = calculateStdDev(bestFitnesses);
            double meanRuntime = Arrays.stream(runtimes).average().orElse(Double.NaN);
            
            System.out.println("======================================");
            System.out.println("Results Summary for " + problemName);
            System.out.println("Mean Fitness: " + meanFitness);
            System.out.println("Std Dev Fitness: " + stdFitness);
            System.out.println("Mean Runtime: " + meanRuntime + " ms");
            System.out.println("======================================\n");
        }
    }
    
    /**
     * 標準偏差を計算するヘルパーメソッド
     */
    private static double calculateStdDev(double[] values) {
        double mean = Arrays.stream(values).average().orElse(Double.NaN);
        double temp = 0;
        for (double a : values) {
            temp += (a - mean) * (a - mean);
        }
        return Math.sqrt(temp / values.length);
    }
}