package efwa;

import optimization.OptimizationProblem;
import optimization.benchmarks.SphereFunction;
import optimization.benchmarks.RosenbrockFunction;
import optimization.benchmarks.AckleyFunction;
import optimization.benchmarks.RastriginFunction;
import utils.StatisticsUtils;
import java.util.Arrays;

public class EFWAApplication {
    
    public static void main(String[] args) {
        // 実験パラメータ
        int dimension = 20;
        int runs = 30;
        int maxEvaluations = dimension*10000;
        
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
            "Rastrigin",
            "Rosenbrock",
            "Ackley"
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
            double[] runtimes = new double[runs];
            
            // 複数回の実行
            for (int r = 0; r < runs; r++) {
                // System.out.println("Run " + (r + 1) + "/" + runs);
                
                // EFWAのインスタンス作成
                EFWA efwa = new EFWA(problem);
                efwa.setParameters(
                    5,           // populationSize
                    maxEvaluations, // maxEvaluations
                    40.0,        // explosionAmplitude
                    50,          // maxExplosionSparks
                    0.04,        // boundingCoeffA
                    0.8,         // boundingCoeffB
                    5,           // gaussianSparks
                    0.8,         // initialMinAmplitude (will be scaled by dimension range)
                    0.001,       // finalMinAmplitude (will be scaled by dimension range)
                    true         // useNonLinearDecrease
                );
                
                // アルゴリズムの実行と時間計測
                long startTime = System.currentTimeMillis();
                double[] solution = efwa.optimize();
                long endTime = System.currentTimeMillis();
                
                // 結果の保存
                double fitness = problem.evaluate(solution);
                bestFitnesses[r] = fitness;
                runtimes[r] = endTime - startTime;
                
                // System.out.println("Best fitness: " + fitness);
                // System.out.println("Runtime: " + runtimes[r] + " ms");
                // System.out.println("Best position: " + Arrays.toString(solution));
            }
            
            // 統計の計算と表示
            double meanFitness = StatisticsUtils.calculateMean(bestFitnesses);
            double stdFitness = StatisticsUtils.calculateStdDev(bestFitnesses);
            double meanRuntime = StatisticsUtils.calculateMedian(runtimes);
            
            System.out.println("======================================");
            System.out.println("Results Summary for " + problemName);
            System.out.println("Mean Fitness: " + meanFitness);
            System.out.println("Std Dev Fitness: " + stdFitness);
            System.out.println("Mean Runtime: " + meanRuntime + " ms");
            System.out.println("======================================\n");
        }
    }
}