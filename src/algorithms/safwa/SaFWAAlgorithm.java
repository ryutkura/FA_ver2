package safwa;

import optimization.OptimizationProblem;
import java.util.ArrayList;
import java.util.Arrays;
// import java.util.Comparator;
import java.util.List;
import java.util.Random;


/**
 * Self-Adaptive Fireworks Algorithm (SaFWA) の実装
 * 参考文献: "A Self-Adaptive Fireworks Algorithm for Classification Problems"
 */
public class SaFWAAlgorithm {
    // アルゴリズムのパラメータ
    private int populationSize;         // 花火の数 n
    private int maxEvaluations;         // 最大評価回数
    private int totalSparks;            // 総火花数 m
    private double boundingCoeffA;      // 下限係数 a
    private double boundingCoeffB;      // 上限係数 b
    private double maxAmplitude;        // 最大爆発振幅 Â
    private int gaussianSparks;         // ガウス火花の数
    private int learningPeriod;         // 学習期間 LP
    
    // 自己適応メカニズム用のパラメータ
    private int strategyNum;            // 戦略数
    private double[] strategyProbs;     // 戦略選択確率 P
    private int[] strategySuccessFlags; // 戦略成功フラグ
    private int[] strategyFailureFlags; // 戦略失敗フラグ
    private int[][] totalSuccessFlags;  // 累積成功フラグ
    private int[][] totalFailureFlags;  // 累積失敗フラグ
    
    // 問題と状態変数
    private OptimizationProblem problem;
    private int currentGeneration;
    private double[] bestSolution;
    private double bestFitness;
    private Random random;
    
    /**
     * コンストラクタ
     */
    public SaFWAAlgorithm(OptimizationProblem problem) {
        this.problem = problem;
        
        // デフォルトパラメータの設定
        this.populationSize = 10;
        this.maxEvaluations = 100000;
        this.totalSparks = 90;
        this.boundingCoeffA = 0.04;
        this.boundingCoeffB = 0.8;
        this.maxAmplitude = 2.0;
        this.gaussianSparks = 8;
        this.learningPeriod = 10;
        
        // 自己適応メカニズムの初期化
        this.strategyNum = 4;
        this.strategyProbs = new double[strategyNum];
        this.strategySuccessFlags = new int[strategyNum];
        this.strategyFailureFlags = new int[strategyNum];
        this.totalSuccessFlags = new int[learningPeriod][strategyNum];
        this.totalFailureFlags = new int[learningPeriod][strategyNum];
        
        // 各戦略の初期確率を均等に設定
        Arrays.fill(strategyProbs, 1.0 / strategyNum);
        
        this.random = new Random();
        this.currentGeneration = 0;
    }
    
    /**
     * パラメータを設定するメソッド
     */
    public void setParameters(int populationSize, int maxEvaluations, int totalSparks, double boundingCoeffA, 
                             double boundingCoeffB, double maxAmplitude, int gaussianSparks,
                             int learningPeriod) {
        this.populationSize = populationSize;
        this.maxEvaluations = maxEvaluations;
        this.totalSparks = totalSparks;
        this.boundingCoeffA = boundingCoeffA;
        this.boundingCoeffB = boundingCoeffB;
        this.maxAmplitude = maxAmplitude;
        this.gaussianSparks = gaussianSparks;
        this.learningPeriod = learningPeriod;
        
        // 学習期間が変更された場合は配列を再初期化
        if (learningPeriod != this.learningPeriod) {
            this.totalSuccessFlags = new int[learningPeriod][strategyNum];
            this.totalFailureFlags = new int[learningPeriod][strategyNum];
        }
    }
    
    /**
     * 最大評価回数を設定するメソッド
     */
    public void setMaxEvaluations(int maxEvaluations) {
        this.maxEvaluations = maxEvaluations;
    }

    /**
     * 最適化の実行
     */
    public double[] optimize() {
        int dimension = problem.getDimension();
        double[] lowerBounds = problem.getLowerBounds();
        double[] upperBounds = problem.getUpperBounds();
        
        // 評価カウンタと世代カウンタの初期化
        int evalCounter = 0;
        currentGeneration = 0;
        
        // 結果保存用の変数
        bestFitness = Double.MAX_VALUE;
        bestSolution = new double[dimension];
        
        // 花火の初期集団を生成
        List<Firework> fireworks = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            double[] position = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                position[j] = lowerBounds[j] + random.nextDouble() * (upperBounds[j] - lowerBounds[j]);
            }
            double fitness = problem.evaluate(position);
            evalCounter++;
            
            fireworks.add(new Firework(position, fitness));
            
            // 最良解の更新
            if (fitness < bestFitness) {
                bestFitness = fitness;
                System.arraycopy(position, 0, bestSolution, 0, dimension);
            }
        }
        
        // メインループ
        while (evalCounter < maxEvaluations) {
            // すべての生成された解（花火、爆発火花、ガウス火花、戦略ベースの子孫）
            List<Firework> allSolutions = new ArrayList<>(fireworks);
            
            // ----- 爆発フェーズ -----
            double minFitness = Double.MAX_VALUE;
            double maxFitness = Double.MIN_VALUE;
            
            // 最小/最大適応度の計算
            for (Firework firework : fireworks) {
                if (firework.fitness < minFitness) minFitness = firework.fitness;
                if (firework.fitness > maxFitness) maxFitness = firework.fitness;
            }
            
            // 各花火について爆発火花を生成
            for (Firework firework : fireworks) {
                // 火花数の計算 (式1)
                int sparkCount = calculateSparkCount(firework.fitness, maxFitness, minFitness, fireworks);
                
                // 爆発振幅の計算 (式2)
                double amplitude = calculateAmplitude(firework.fitness, minFitness, fireworks);
                
                // 爆発火花の生成と評価
                for (int j = 0; j < sparkCount && evalCounter < maxEvaluations; j++) {
                    double[] sparkPosition = generateExplosionSpark(firework.position, amplitude, lowerBounds, upperBounds);
                    double sparkFitness = problem.evaluate(sparkPosition);
                    evalCounter++;
                    
                    Firework spark = new Firework(sparkPosition, sparkFitness);
                    allSolutions.add(spark);
                    
                    // 最良解の更新
                    if (sparkFitness < bestFitness) {
                        bestFitness = sparkFitness;
                        System.arraycopy(sparkPosition, 0, bestSolution, 0, dimension);
                    }
                }
            }
            
            // ----- ガウス変異フェーズ -----

            // for (int i = 0; i < gaussianSparks && evalCounter < maxEvaluations; i++) {
            //     // ランダムに花火を選択
            //     Firework firework = fireworks.get(random.nextInt(fireworks.size()));
                
            //     // ガウス変異火花を生成
            //     double[] gaussianSparkPosition = generateGaussianSpark(firework.position, bestSolution, lowerBounds, upperBounds);
            //     double gaussianSparkFitness = problem.evaluate(gaussianSparkPosition);
            //     evalCounter++;
                
            //     Firework gaussianSpark = new Firework(gaussianSparkPosition, gaussianSparkFitness);
            //     allSolutions.add(gaussianSpark);
                
            //     // 最良解の更新
            //     if (gaussianSparkFitness < bestFitness) {
            //         bestFitness = gaussianSparkFitness;
            //         System.arraycopy(gaussianSparkPosition, 0, bestSolution, 0, dimension);
            //     }
            // }これが元ネタ
            /// ★ 修正: 各花火から 1 本ずつ (= populationSize 本)
            for (Firework firework : fireworks) {
                for (int g = 0; g < 1 && evalCounter<maxEvaluations; g++) {

                double[] gaussianSparkPosition =
                        generateGaussianSpark(firework.position, lowerBounds, upperBounds);
                double gaussianSparkFitness = problem.evaluate(gaussianSparkPosition);
                evalCounter++;

                allSolutions.add(new Firework(gaussianSparkPosition, gaussianSparkFitness));
                }
            }
            
            // ----- 戦略ベースの子孫生成 -----
            for (int i = 0; i < populationSize && evalCounter < maxEvaluations; i++) {
                Firework current = fireworks.get(i);
                
                // ルーレット選択で戦略を選択
                int curStrategy = selectStrategyByRoulette();
                
                // 選択された戦略で子孫を生成
                double[] offspringPosition = generateOffspring(curStrategy, current.position, fireworks, bestSolution, dimension);
                
                // 境界チェック
                for (int j = 0; j < dimension; j++) {
                    if (offspringPosition[j] < lowerBounds[j]) {
                        offspringPosition[j] = lowerBounds[j];
                    } else if (offspringPosition[j] > upperBounds[j]) {
                        offspringPosition[j] = upperBounds[j];
                    }
                }
                
                // 子孫の評価
                double offspringFitness = problem.evaluate(offspringPosition);
                evalCounter++;
                
                // 成功/失敗フラグの更新
                if (offspringFitness < current.fitness) {
                    // 成功: 現在の花火を置き換え
                    current.position = offspringPosition;
                    current.fitness = offspringFitness;
                    
                    // 成功フラグの更新
                    strategySuccessFlags[curStrategy]++;
                } else {
                    // 失敗フラグの更新
                    strategyFailureFlags[curStrategy]++;
                }
                
                // 最良解の更新
                if (offspringFitness < bestFitness) {
                    bestFitness = offspringFitness;
                    System.arraycopy(offspringPosition, 0, bestSolution, 0, dimension);
                }
            }
            
            // ----- 自己適応メカニズム -----
            if (currentGeneration % learningPeriod == 0) {
                // 成功/失敗フラグの記録
                totalSuccessFlags[currentGeneration % learningPeriod] = strategySuccessFlags.clone();
                totalFailureFlags[currentGeneration % learningPeriod] = strategyFailureFlags.clone();
                
                // 戦略選択確率の更新
                updateStrategyProbabilities();
                
                // フラグのリセット
                Arrays.fill(strategySuccessFlags, 0);
                Arrays.fill(strategyFailureFlags, 0);
            }
            
            // ----- 次世代選択 -----
            // グローバルベストを保存
            Firework bestFirework = findBestFirework(allSolutions);
            
            // 距離ベースの選択で残りのn-1個体を選択
            fireworks = selectNextGeneration(allSolutions, bestFirework);
            
            currentGeneration++;
        }
        
        return bestSolution;
    }

    /**
     * 最良適応度の取得
     */
    public double getBestFitness() {
        return bestFitness;
    }

    private static double clamp(double v, double lo, double hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }
    
    /**
     * 火花数の計算 (式1)
     */
    private int calculateSparkCount(double fitness, double maxFitness, double minFitness, List<Firework> fireworks) {
        double eps = 1e-10; // 0除算防止
        
        double numerator = maxFitness - fitness + eps;
        double denominator = 0;
        for (Firework firework : fireworks) {
            denominator += (maxFitness - firework.fitness);
        }
        denominator += eps;
        
        int sparkCount = (int) Math.round(totalSparks * (numerator / denominator));
        
        // 制限
        if (sparkCount < boundingCoeffA * totalSparks) {
            sparkCount = (int) Math.round(boundingCoeffA * totalSparks);
        } else if (sparkCount > boundingCoeffB * totalSparks) {
            sparkCount = (int) Math.round(boundingCoeffB * totalSparks);
        }
        
        return sparkCount;
    }
    
    /**
     * 爆発振幅の計算 (式2)
     */
    private double calculateAmplitude(double fitness, double minFitness, List<Firework> fireworks) {
        double eps = 1e-10; // 0除算防止
        
        double numerator = fitness - minFitness + eps;
        double denominator = 0;
        for (Firework firework : fireworks) {
            denominator += (firework.fitness - minFitness);
        }
        denominator += eps;
        
        return maxAmplitude * (numerator / denominator);
    }
    
    /**
     * 爆発火花の生成
     */
    private double[] generateExplosionSpark(double[] position, double amplitude, double[] lowerBounds, double[] upperBounds) {
        double[] sparkPosition = position.clone();
        for (int k = 0; k < sparkPosition.length; k++) {
            double delta = (random.nextDouble()*2-1) * amplitude;
            sparkPosition[k] = clamp(sparkPosition[k] + delta, lowerBounds[k], upperBounds[k]);
        }
        
        return sparkPosition;
    }
    
    /**
     * ガウス変異火花の生成
     */
    // private double[] generateGaussianSpark(double[] position, double[] bestPosition, double[] lowerBounds, double[] upperBounds) {
    //     int dimension = position.length;
    //     double[] sparkPosition = position.clone();
        
    //     // ガウス変異を適用する次元をランダムに選択
    //     int numDimToChange = 1 + random.nextInt(dimension);
    //     for (int i = 0; i < numDimToChange; i++) {
    //         int dimIdx = random.nextInt(dimension);
            
    //         // ガウス変異: 現在の位置と最良位置の差に基づく変異
    //         double delta = random.nextGaussian() * (bestPosition[dimIdx] - position[dimIdx]);
    //         sparkPosition[dimIdx] += delta;
            
    //         // 境界チェック
    //         if (sparkPosition[dimIdx] < lowerBounds[dimIdx]) {
    //             sparkPosition[dimIdx] = lowerBounds[dimIdx];
    //         } else if (sparkPosition[dimIdx] > upperBounds[dimIdx]) {
    //             sparkPosition[dimIdx] = upperBounds[dimIdx];
    //         }
    //     }
        
    //     return sparkPosition;
    // }

    /**
     * Gaussian‑mutation spark (論文の「ガウス分布で広がる一点」)
     */
    private double[] generateGaussianSpark(double[] position,
                                        double[] lowerBounds,
                                        double[] upperBounds) {      // ★ bestPosition 引数を削除
        double[] spark = position.clone();
        for (int k = 0; k < spark.length; k++) {
            double delta = random.nextGaussian() * 0.1 * (upperBounds[k]-lowerBounds[k]);
            spark[k] = clamp(spark[k] + delta, lowerBounds[k], upperBounds[k]);
        }
        return spark;
    }
    
    /**
     * ルーレット選択で戦略を選択
     */
    private int selectStrategyByRoulette() {
        double r = random.nextDouble();
        double cumulativeProb = 0.0;
        
        for (int i = 0; i < strategyNum; i++) {
            cumulativeProb += strategyProbs[i];
            if (r <= cumulativeProb) {
                return i;
            }
        }
        
        return strategyNum - 1; // デフォルト
    }
    
    /**
     * 戦略ベースの子孫生成
     */
    private double[] generateOffspring(int strategy, double[] current, List<Firework> fireworks, double[] best, int dimension) {
        double[] offspring = new double[dimension];
        double F = 0.5 +  random.nextDouble() * 0.5; // F ∈ [0.5, 1]
        
        switch (strategy) {
            case 0: // CSGS1 - DE/rand/1 (式6)
                {
                    int r1, r2, r3;
                    do {
                        r1 = random.nextInt(fireworks.size());
                    } while (fireworks.get(r1).position == current);
                    
                    do {
                        r2 = random.nextInt(fireworks.size());
                    } while (r2 == r1 || fireworks.get(r2).position == current);
                    
                    do {
                        r3 = random.nextInt(fireworks.size());
                    } while (r3 == r1 || r3 == r2 || fireworks.get(r3).position == current);
                    
                    double[] xr1 = fireworks.get(r1).position;
                    double[] xr2 = fireworks.get(r2).position;
                    double[] xr3 = fireworks.get(r3).position;
                    
                    for (int i = 0; i < dimension; i++) {
                        offspring[i] = xr1[i] + F * (xr2[i] - xr3[i]);
                    }
                }
                break;
                
            case 1: // CSGS2 - DE/rand/2 (式7)
                {
                    int r1, r2, r3, r4, r5;
                    do {
                        r1 = random.nextInt(fireworks.size());
                    } while (fireworks.get(r1).position == current);
                    
                    do {
                        r2 = random.nextInt(fireworks.size());
                    } while (r2 == r1 || fireworks.get(r2).position == current);
                    
                    do {
                        r3 = random.nextInt(fireworks.size());
                    } while (r3 == r1 || r3 == r2 || fireworks.get(r3).position == current);
                    
                    do {
                        r4 = random.nextInt(fireworks.size());
                    } while (r4 == r1 || r4 == r2 || r4 == r3 || fireworks.get(r4).position == current);
                    
                    do {
                        r5 = random.nextInt(fireworks.size());
                    } while (r5 == r1 || r5 == r2 || r5 == r3 || r5 == r4 || fireworks.get(r5).position == current);
                    
                    double[] xr1 = fireworks.get(r1).position;
                    double[] xr2 = fireworks.get(r2).position;
                    double[] xr3 = fireworks.get(r3).position;
                    double[] xr4 = fireworks.get(r4).position;
                    double[] xr5 = fireworks.get(r5).position;
                    
                    for (int i = 0; i < dimension; i++) {
                        offspring[i] = xr1[i] + F * (xr2[i] - xr3[i]) + F * (xr4[i] - xr5[i]);
                    }
                }
                break;
                
            case 2: // CSGS3 - DE/best/2 (式8)
                {
                    int r1, r2, r3, r4;
                    do {
                        r1 = random.nextInt(fireworks.size());
                    } while (fireworks.get(r1).position == current);
                    
                    do {
                        r2 = random.nextInt(fireworks.size());
                    } while (r2 == r1 || fireworks.get(r2).position == current);
                    
                    do {
                        r3 = random.nextInt(fireworks.size());
                    } while (r3 == r1 || r3 == r2 || fireworks.get(r3).position == current);
                    
                    do {
                        r4 = random.nextInt(fireworks.size());
                    } while (r4 == r1 || r4 == r2 || r4 == r3 || fireworks.get(r4).position == current);
                    
                    double[] xr1 = fireworks.get(r1).position;
                    double[] xr2 = fireworks.get(r2).position;
                    double[] xr3 = fireworks.get(r3).position;
                    double[] xr4 = fireworks.get(r4).position;
                    
                    for (int i = 0; i < dimension; i++) {
                        offspring[i] = best[i] + F * (xr1[i] - xr2[i]) + F * (xr3[i] - xr4[i]);
                    }
                }
                break;
                
            case 3: // CSGS4 - DE/current-to-best/2 (式9)
                {
                    int r1, r2, r3, r4;
                    do {
                        r1 = random.nextInt(fireworks.size());
                    } while (fireworks.get(r1).position == current);
                    
                    do {
                        r2 = random.nextInt(fireworks.size());
                    } while (r2 == r1 || fireworks.get(r2).position == current);
                    
                    do {
                        r3 = random.nextInt(fireworks.size());
                    } while (r3 == r1 || r3 == r2 || fireworks.get(r3).position == current);
                    
                    do {
                        r4 = random.nextInt(fireworks.size());
                    } while (r4 == r1 || r4 == r2 || r4 == r3 || fireworks.get(r4).position == current);
                    
                    double[] xr1 = fireworks.get(r1).position;
                    double[] xr2 = fireworks.get(r2).position;
                    double[] xr3 = fireworks.get(r3).position;
                    double[] xr4 = fireworks.get(r4).position;
                    
                    for (int i = 0; i < dimension; i++) {
                        offspring[i] = current[i] + F * (best[i] - current[i]) + F * (xr1[i] - xr2[i]) + F * (xr3[i] - xr4[i]);
                    }
                }
                break;
                
            default:
                System.arraycopy(current, 0, offspring, 0, dimension);
                break;
        }
        
        return offspring;
    }
    
    /**
     * 戦略選択確率の更新 (式4-5)
     */
    // private void updateStrategyProbabilities() {
    //     double[] pPrime = new double[strategyNum];
    //     double pPrimeSum = 0.0;
    //     double epsilon = 1e-10; // 0除算防止
        
    //     for (int q = 0; q < strategyNum; q++) {
    //         int successSum = 0;
    //         int totalSum = 0;
            
    //         for (int gen = 0; gen < learningPeriod; gen++) {
    //             successSum += totalSuccessFlags[gen][q];
    //             totalSum += totalSuccessFlags[gen][q] + totalFailureFlags[gen][q];
    //         }
            
    //         if (successSum > 0) {
    //             pPrime[q] = (double) successSum / (totalSum + epsilon);
    //         } else {
    //             // すべての試行が失敗した場合（論文の式4の2番目のケース）
    //             pPrime[q] = epsilon / (totalSum + epsilon);
    //         }
            
    //         pPrimeSum += pPrime[q];
    //     }
        
    //     // 正規化 (式5)
    //     for (int q = 0; q < strategyNum; q++) {
    //         strategyProbs[q] = pPrime[q] / pPrimeSum;
    //     }
    // }
    private void updateStrategyProbabilities() {
    double[] pPrime = new double[strategyNum];
    double epsilon = 1e-10;

        for (int q = 0; q < strategyNum; q++) {
            int successSum = 0;
            int failureSum = 0;

            for (int g = 0; g < learningPeriod; g++) {
                successSum += totalSuccessFlags[g][q];
                failureSum += totalFailureFlags[g][q];
            }

            // --- 式(4) 4 分岐 ---
            if (successSum > 0 && failureSum > 0) {                    // ケース①
                pPrime[q] = (double) successSum / (successSum + failureSum + epsilon);
            } else if (successSum > 0) {                               // ケース②
                pPrime[q] = (double) successSum / (successSum + epsilon);
            } else if (failureSum > 0) {                               // ケース③
                pPrime[q] = epsilon / (failureSum + epsilon);
            } else {                                                   // ケース④ 全く試行されていない
                pPrime[q] = 1.0 / strategyNum;
            }
        }

        // 正規化 (式5)
        double sum = 0.0;
        for (double v : pPrime) sum += v;
        for (int q = 0; q < strategyNum; q++) strategyProbs[q] = pPrime[q] / sum;
    }

    
    /**
     * 最良の個体を見つける
     */
    private Firework findBestFirework(List<Firework> fireworks) {
        Firework best = fireworks.get(0);
        for (Firework firework : fireworks) {
            if (firework.fitness < best.fitness) {
                best = firework;
            }
        }
        return best;
    }
    
    /**
     * 次世代選択 (式3)
     */
    private List<Firework> selectNextGeneration(List<Firework> allSolutions, Firework bestFirework) {
        List<Firework> nextGeneration = new ArrayList<>();
        
        // 最良個体を保存
        nextGeneration.add(bestFirework);
        allSolutions.remove(bestFirework);
        
        // 残りの個体を距離に基づいて選択
        if (allSolutions.size() > 0 && populationSize > 1) {
            double[] distances = new double[allSolutions.size()];
            double distanceSum = 0.0;
            
            // 各個体の他の個体からの平均距離を計算
            for (int i = 0; i < allSolutions.size(); i++) {
                double distanceToOthers = 0.0;
                
                for (int j = 0; j < allSolutions.size(); j++) {
                    if (i != j) {
                        distanceToOthers += calculateDistance(allSolutions.get(i).position, allSolutions.get(j).position);
                    }
                }
                
                distances[i] = distanceToOthers / (allSolutions.size() - 1);
                distanceSum += distances[i];
            }
            
            // ルーレット選択で残りの個体を選ぶ
            for (int i = 1; i < populationSize; i++) {
                double r = random.nextDouble() * distanceSum;
                double cumulativeDistance = 0.0;
                
                for (int j = 0; j < allSolutions.size(); j++) {
                    cumulativeDistance += distances[j];
                    if (cumulativeDistance >= r) {
                        nextGeneration.add(allSolutions.get(j));
                        distanceSum -= distances[j];
                        allSolutions.remove(j);
                        distances = ArrayRemove(distances, j);
                        break;
                    }
                }
                
                // 希少な場合の処理
                if (allSolutions.isEmpty() && nextGeneration.size() < populationSize) {
                    // 最良解のコピーを追加
                    nextGeneration.add(new Firework(bestFirework.position.clone(), bestFirework.fitness));
                }
            }
        } else {
            // 個体が足りない場合は最良解で埋める
            while (nextGeneration.size() < populationSize) {
                nextGeneration.add(new Firework(bestFirework.position.clone(), bestFirework.fitness));
            }
        }
        
        return nextGeneration;
    }
    
    /**
     * 2つの位置間のユークリッド距離を計算
     */
    private double calculateDistance(double[] pos1, double[] pos2) {
        double sum = 0.0;
        for (int i = 0; i < pos1.length; i++) {
            sum += Math.pow(pos1[i] - pos2[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    /**
     * 配列から要素を削除する補助メソッド
     */
    private double[] ArrayRemove(double[] arr, int index) {
        double[] result = new double[arr.length - 1];
        System.arraycopy(arr, 0, result, 0, index);
        System.arraycopy(arr, index + 1, result, index, arr.length - index - 1);
        return result;
    }
    
    /**
     * 花火クラス（個体を表す）
     */
    private class Firework {
        double[] position;
        double fitness;
        
        public Firework(double[] position, double fitness) {
            this.position = position;
            this.fitness = fitness;
        }
    }
}