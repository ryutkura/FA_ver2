package efwa;

import optimization.OptimizationProblem;
import java.util.ArrayList;
// import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Enhanced Fireworks Algorithm のメインクラス
 */
public class EFWA {
    // アルゴリズムのパラメータ
    private int populationSize;        // 花火の数 N
    private int maxEvaluations;        // 最大評価回数
    private double explosionAmplitude; // 爆発振幅の定数 Â
    private int maxExplosionSparks;    // 爆発火花の最大数 Me
    private double boundingCoeffA;     // 下限係数 a
    private double boundingCoeffB;     // 上限係数 b
    private int gaussianSparks;        // ガウス火花の数 Mg
    
    // 最小爆発振幅のパラメータ
    private double initialMinAmplitude;  // 初期最小振幅
    private double finalMinAmplitude;    // 最終最小振幅
    private boolean useNonLinearDecrease; // 非線形減少を使用するかどうか
    
    // 問題と状態変数
    private OptimizationProblem problem;
    private int currentEvaluations;
    private double[] bestPosition;
    private double bestFitness;
    
    /**
     * Enhanced Fireworks Algorithm のコンストラクタ
     */
    public EFWA(OptimizationProblem problem) {
        this.problem = problem;
        
        // デフォルトパラメータの設定（論文の推奨値に基づく）
        this.populationSize = 5;
        this.maxEvaluations = 300000;
        this.explosionAmplitude = 40.0;
        this.maxExplosionSparks = 50;
        this.boundingCoeffA = 0.04;
        this.boundingCoeffB = 0.8;
        this.gaussianSparks = 5;
        
        // 最小爆発振幅のパラメータ
        double[] lowerBounds = problem.getLowerBounds();
        double[] upperBounds = problem.getUpperBounds();
        double avgRange = 0;
        for (int i = 0; i < problem.getDimension(); i++) {
            avgRange += (upperBounds[i] - lowerBounds[i]);
        }
        avgRange /= problem.getDimension();
        
        this.initialMinAmplitude = 0.8 * avgRange;
        this.finalMinAmplitude = 0.001 * avgRange;
        this.useNonLinearDecrease = true;
    }
    
    /**
     * アルゴリズムのパラメータを設定するメソッド
     */
    public void setParameters(int populationSize, int maxEvaluations, double explosionAmplitude,
                             int maxExplosionSparks, double boundingCoeffA, double boundingCoeffB,
                             int gaussianSparks, double initialMinAmplitude, double finalMinAmplitude,
                             boolean useNonLinearDecrease) {
        this.populationSize = populationSize;
        this.maxEvaluations = maxEvaluations;
        this.explosionAmplitude = explosionAmplitude;
        this.maxExplosionSparks = maxExplosionSparks;
        this.boundingCoeffA = boundingCoeffA;
        this.boundingCoeffB = boundingCoeffB;
        this.gaussianSparks = gaussianSparks;
        this.initialMinAmplitude = initialMinAmplitude;
        this.finalMinAmplitude = finalMinAmplitude;
        this.useNonLinearDecrease = useNonLinearDecrease;
    }
    
    /**
     * 最適化を実行するメソッド
     */
    public double[] optimize() {
        int dimension = problem.getDimension();
        double[] lowerBounds = problem.getLowerBounds();
        double[] upperBounds = problem.getUpperBounds();
        
        // 初期化
        currentEvaluations = 0;
        bestFitness = Double.MAX_VALUE;
        bestPosition = new double[dimension];
        
        // 花火の初期集団を生成
        Firework[] fireworks = new Firework[populationSize];
        for (int i = 0; i < populationSize; i++) {
            double[] position = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                position[j] = lowerBounds[j] + Math.random() * (upperBounds[j] - lowerBounds[j]);
            }
            double fitness = problem.evaluate(position);
            currentEvaluations++;
            
            fireworks[i] = new Firework(position, fitness);
            
            // 最良解の更新
            if (fitness < bestFitness) {
                bestFitness = fitness;
                System.arraycopy(position, 0, bestPosition, 0, dimension);
            }
        }
        
        // メインループ
        while (currentEvaluations < maxEvaluations) {
            // 爆発振幅と火花数の計算
            calculateAmplitudeAndSparks(fireworks);
            
            // すべての花火からの火花を保持するリスト
            List<Firework> allSparks = new ArrayList<>();
            
            // 元の花火も候補に含める
            for (Firework firework : fireworks) {
                allSparks.add(firework);
            }
            
            // 各花火について爆発火花とガウス火花を生成
            for (int i = 0; i < populationSize; i++) {
                Firework firework = fireworks[i];
                
                // 爆発火花の生成
                for (int j = 0; j < firework.numSparks; j++) {
                    if (currentEvaluations >= maxEvaluations) break;
                    
                    double[] sparkPosition = generateExplosionSpark(firework.position, firework.amplitude, lowerBounds, upperBounds);
                    double sparkFitness = problem.evaluate(sparkPosition);
                    currentEvaluations++;
                    
                    allSparks.add(new Firework(sparkPosition, sparkFitness));
                    
                    // 最良解の更新
                    if (sparkFitness < bestFitness) {
                        bestFitness = sparkFitness;
                        System.arraycopy(sparkPosition, 0, bestPosition, 0, dimension);
                    }
                }
            }
            
            // ガウス火花の生成
            for (int i = 0; i < gaussianSparks; i++) {
                if (currentEvaluations >= maxEvaluations) break;
                
                // ランダムに花火を選択
                Firework firework = fireworks[(int)(Math.random() * populationSize)];
                
                double[] sparkPosition = generateGaussianSpark(firework.position, bestPosition, lowerBounds, upperBounds);
                double sparkFitness = problem.evaluate(sparkPosition);
                currentEvaluations++;
                
                allSparks.add(new Firework(sparkPosition, sparkFitness));
                
                // 最良解の更新
                if (sparkFitness < bestFitness) {
                    bestFitness = sparkFitness;
                    System.arraycopy(sparkPosition, 0, bestPosition, 0, dimension);
                }
            }
            
            // 次世代の選択（エリート＋ランダム選択）
            fireworks = selectNextGeneration(allSparks);
            
            // // 進捗の表示（オプション）
            // if (currentEvaluations % 10000 == 0) {
            //     System.out.println("Evaluations: " + currentEvaluations + ", Best fitness: " + bestFitness);
            // }
        }
        
        return bestPosition;
    }
    
    /**
     * 各花火の爆発振幅と火花数を計算するメソッド
     */
    private void calculateAmplitudeAndSparks(Firework[] fireworks) {
        double sumFitness = 0;
        double minFitness = Double.MAX_VALUE;
        double maxFitness = Double.MIN_VALUE;
        
        // 最小/最大適応度と合計を計算
        for (Firework firework : fireworks) {
            if (firework.fitness < minFitness) {
                minFitness = firework.fitness;
            }
            if (firework.fitness > maxFitness) {
                maxFitness = firework.fitness;
            }
            sumFitness += firework.fitness;
        }
        
        double epsilon = 1e-10; // 0除算防止用の小さな値
        
        // 最小爆発振幅の計算（時間依存）
        double minAmplitude = calculateMinimalAmplitude();
        
        // 各花火の爆発振幅と火花数を計算
        for (Firework firework : fireworks) {
            // 爆発振幅の計算
            firework.amplitude = explosionAmplitude * 
                                (firework.fitness - minFitness + epsilon) / 
                                (sumFitness - minFitness * populationSize + epsilon);
            
            // 最小爆発振幅のチェック
            for (int k = 0; k < problem.getDimension(); k++) {
                double range = problem.getUpperBounds()[k] - problem.getLowerBounds()[k];
                double amplitudeForDimension = firework.amplitude * range;
                if (amplitudeForDimension < minAmplitude) {
                    firework.amplitude = minAmplitude / range;
                }
            }
            
            // 火花数の計算
            firework.numSparks = (int) Math.round(maxExplosionSparks * 
                                                 (maxFitness - firework.fitness + epsilon) / 
                                                 (populationSize * maxFitness - sumFitness + epsilon));
            
            // 火花数の制限
            if (firework.numSparks < boundingCoeffA * maxExplosionSparks) {
                firework.numSparks = (int) Math.round(boundingCoeffA * maxExplosionSparks);
            } else if (firework.numSparks > boundingCoeffB * maxExplosionSparks) {
                firework.numSparks = (int) Math.round(boundingCoeffB * maxExplosionSparks);
            }
        }
    }
    
    /**
     * 最小爆発振幅を計算するメソッド（線形または非線形減少）
     */
    private double calculateMinimalAmplitude() {
        if (useNonLinearDecrease) {
            // 非線形減少
            return initialMinAmplitude - (initialMinAmplitude - finalMinAmplitude) / maxEvaluations * 
                   Math.sqrt((2.0 * maxEvaluations - currentEvaluations) * currentEvaluations);
        } else {
            // 線形減少
            return initialMinAmplitude - (initialMinAmplitude - finalMinAmplitude) / maxEvaluations * currentEvaluations;
        }
    }
    
    /**
     * 爆発火花を生成するメソッド（EFWA方式）
     */
    private double[] generateExplosionSpark(double[] position, double amplitude, double[] lowerBounds, double[] upperBounds) {
        int dimension = position.length;
        double[] sparkPosition = new double[dimension];
        System.arraycopy(position, 0, sparkPosition, 0, dimension);
        
        // 各次元でランダムに選択し、変位を適用
        for (int k = 0; k < dimension; k++) {
            // 50%の確率で次元を選択
            if (Math.random() < 0.5) {
                // 各次元ごとに異なる変位を計算
                double displacement = amplitude * (upperBounds[k] - lowerBounds[k]) * (Math.random() * 2 - 1);
                sparkPosition[k] += displacement;
                
                // 探索空間外の場合、新しいマッピング演算子を適用
                if (sparkPosition[k] < lowerBounds[k] || sparkPosition[k] > upperBounds[k]) {
                    sparkPosition[k] = lowerBounds[k] + Math.random() * (upperBounds[k] - lowerBounds[k]);
                }
            }
        }
        
        return sparkPosition;
    }
    
    /**
     * ガウス火花を生成するメソッド（EFWA方式）
     */
    private double[] generateGaussianSpark(double[] position, double[] bestPosition, double[] lowerBounds, double[] upperBounds) {
        int dimension = position.length;
        double[] sparkPosition = new double[dimension];
        System.arraycopy(position, 0, sparkPosition, 0, dimension);
        
        // ガウス分布からのランダム値
        double e = new Random().nextGaussian();
        
        // 各次元でランダムに選択し、変位を適用
        for (int k = 0; k < dimension; k++) {
            // 50%の確率で次元を選択
            if (Math.random() < 0.5) {
                // 最良解方向への伸縮
                sparkPosition[k] = sparkPosition[k] + (bestPosition[k] - sparkPosition[k]) * e;
                
                // 探索空間外の場合、一様ランダムマッピングを適用
                if (sparkPosition[k] < lowerBounds[k] || sparkPosition[k] > upperBounds[k]) {
                    sparkPosition[k] = lowerBounds[k] + Math.random() * (upperBounds[k] - lowerBounds[k]);
                }
            }
        }
        
        return sparkPosition;
    }
    
    /**
     * 次世代の選択（エリート＋ランダム選択）
     */
    private Firework[] selectNextGeneration(List<Firework> allSparks) {
        Firework[] nextGeneration = new Firework[populationSize];
        
        // 適応度でソート
        Collections.sort(allSparks, Comparator.comparingDouble(f -> f.fitness));
        
        // 最良個体を保持
        nextGeneration[0] = allSparks.get(0);
        
        // 残りはランダム選択
        List<Firework> candidates = new ArrayList<>(allSparks.subList(1, allSparks.size()));
        for (int i = 1; i < populationSize; i++) {
            if (candidates.isEmpty()) break;
            int randomIndex = (int)(Math.random() * candidates.size());
            nextGeneration[i] = candidates.get(randomIndex);
            candidates.remove(randomIndex);
        }
        
        return nextGeneration;
    }
    
    /**
     * 花火を表すクラス
     */
    private class Firework {
        double[] position;
        double fitness;
        double amplitude;
        int numSparks;
        
        public Firework(double[] position, double fitness) {
            this.position = position;
            this.fitness = fitness;
        }
    }
}
