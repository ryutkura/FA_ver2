package pso;

import optimization.OptimizationProblem;
import java.util.Random;

public class PSO {
    // アルゴリズムのパラメータ
    private int swarmSize;           // 粒子の数
    private int maxIterations;       // 最大反復回数
    private double inertiaWeight;    // 慣性重み
    private double c1;               // 認知パラメータ
    private double c2;               // 社会的パラメータ
    
    // 問題と状態変数
    private OptimizationProblem problem;
    private int currentIteration;
    private double[] globalBestPosition;
    private double globalBestFitness;
    
    // 乱数生成器
    private Random random;
    
    /**
     * Particle Swarm Optimization のコンストラクタ
     */
    public PSO(OptimizationProblem problem) {
        this.problem = problem;
        this.random = new Random();
        
        // デフォルトパラメータの設定
        this.swarmSize = 30;
        this.maxIterations = 10000;
        this.inertiaWeight = 0.729;
        this.c1 = 1.49445;
        this.c2 = 1.49445;
    }
    
    /**
     * アルゴリズムのパラメータを設定するメソッド
     */
    public void setParameters(int swarmSize, int maxIterations, double inertiaWeight, double c1, double c2) {
        this.swarmSize = swarmSize;
        this.maxIterations = maxIterations;
        this.inertiaWeight = inertiaWeight;
        this.c1 = c1;
        this.c2 = c2;
    }
    
    /**
     * 最適化を実行するメソッド
     */
    public double[] optimize() {
        int dimension = problem.getDimension();
        double[] lowerBounds = problem.getLowerBounds();
        double[] upperBounds = problem.getUpperBounds();
        
        // 初期化
        currentIteration = 0;
        globalBestFitness = Double.MAX_VALUE;
        globalBestPosition = new double[dimension];
        
        // 粒子群の初期化
        Particle[] particles = new Particle[swarmSize];
        for (int i = 0; i < swarmSize; i++) {
            // 位置と速度を初期化
            double[] position = new double[dimension];
            double[] velocity = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                // ランダムな位置を生成
                position[j] = lowerBounds[j] + random.nextDouble() * (upperBounds[j] - lowerBounds[j]);
                // 速度を初期化（通常は[-vmax, vmax]の範囲内）
                double vmax = 0.1 * (upperBounds[j] - lowerBounds[j]);
                velocity[j] = -vmax + random.nextDouble() * 2 * vmax;
            }
            
            // 粒子の適応度を評価
            double fitness = problem.evaluate(position);
            
            // 粒子を作成
            particles[i] = new Particle(position, velocity, fitness, position.clone(), fitness);
            
            // グローバルベストの更新
            if (fitness < globalBestFitness) {
                globalBestFitness = fitness;
                System.arraycopy(position, 0, globalBestPosition, 0, dimension);
            }
        }
        
        // メインループ
        while (currentIteration < maxIterations) {
            // 各粒子を更新
            for (int i = 0; i < swarmSize; i++) {
                Particle p = particles[i];
                
                // 速度を更新
                for (int j = 0; j < dimension; j++) {
                    // r1, r2 は [0, 1] の乱数
                    double r1 = random.nextDouble();
                    double r2 = random.nextDouble();
                    
                    // 速度更新式
                    p.velocity[j] = inertiaWeight * p.velocity[j] 
                                   + c1 * r1 * (p.personalBestPosition[j] - p.position[j])
                                   + c2 * r2 * (globalBestPosition[j] - p.position[j]);
                    
                    // 速度の制限（オプション）
                    double vmax = 0.1 * (upperBounds[j] - lowerBounds[j]);
                    if (p.velocity[j] > vmax) p.velocity[j] = vmax;
                    if (p.velocity[j] < -vmax) p.velocity[j] = -vmax;
                }
                
                // 位置を更新
                for (int j = 0; j < dimension; j++) {
                    p.position[j] += p.velocity[j];
                    
                    // // 境界チェック
                    // if (p.position[j] < lowerBounds[j]) {
                    //     p.position[j] = lowerBounds[j];
                    //     p.velocity[j] *= -0.5; // 壁に跳ね返る（オプション）
                    // }
                    // if (p.position[j] > upperBounds[j]) {
                    //     p.position[j] = upperBounds[j];
                    //     p.velocity[j] *= -0.5; // 壁に跳ね返る（オプション）
                    // }
                }
                
                // 新しい位置の適応度を評価
                double fitness = problem.evaluate(p.position);
                p.fitness = fitness;
                
                // パーソナルベストの更新
                if (fitness < p.personalBestFitness) {
                    p.personalBestFitness = fitness;
                    System.arraycopy(p.position, 0, p.personalBestPosition, 0, dimension);
                    
                    // グローバルベストの更新
                    if (fitness < globalBestFitness) {
                        globalBestFitness = fitness;
                        System.arraycopy(p.position, 0, globalBestPosition, 0, dimension);
                    }
                }
            }
            
            // 進捗の表示（オプション）
            // if (currentIteration % 100 == 0) {
            //     System.out.println("Iteration: " + currentIteration + ", Best fitness: " + globalBestFitness);
            // }
            
            currentIteration++;
        }
        
        return globalBestPosition;
    }
    
    /**
     * 粒子を表すクラス
     */
    private class Particle {
        double[] position;              // 現在の位置
        double[] velocity;              // 現在の速度
        double fitness;                 // 現在の適応度
        double[] personalBestPosition;  // パーソナルベスト位置
        double personalBestFitness;     // パーソナルベスト適応度
        
        public Particle(double[] position, double[] velocity, double fitness,
                       double[] personalBestPosition, double personalBestFitness) {
            this.position = position;
            this.velocity = velocity;
            this.fitness = fitness;
            this.personalBestPosition = personalBestPosition;
            this.personalBestFitness = personalBestFitness;
        }
    }
}