/**
 * Project          : aol-ml
 * File             : FactorizationMachine
 * Author           : Nikhil J Joshi (nikhil.joshi@teamaol.com)
 * Date Created     : 10/24/15
 *
 * Descriptor       :
 * " The class usage goes here. "
 *
 */

package org.apache.spark.ml.classification

import scala.collection.mutable

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}

import com.github.fommil.netlib.F2jBLAS

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkException, Logging}
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.param.shared._
import org.apache.spark.mllib.linalg._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.functions.{lit, col}
import org.apache.spark.storage.StorageLevel

import org.apache.spark.mllib.frommaster.MultivariateOnlineSummarizer
import org.apache.spark.mllib.frommaster.{MultiClassSummarizer => NewMultiClassSummarizer}

import scala.util.Random

/**
 * Params for Factorization Machine.
 */
private[classification] trait FactorizationMachineParams extends ProbabilisticClassifierParams
with HasRegParam with HasElasticNetParam with HasMaxIter with HasFitIntercept with HasTol
with HasStandardization with HasWeightCol with HasThreshold {

  /**
   * Set threshold in binary classification, in range [0, 1].
   *
   * If the estimated probability of class label 1 is > threshold, then predict 1, else 0.
   * A high threshold encourages the model to predict 0 more often;
   * a low threshold encourages the model to predict 1 more often.
   *
   * Note: Calling this with threshold p is equivalent to calling `setThresholds(Array(1-p, p))`.
   *       When [[setThreshold()]] is called, any user-set value for [[thresholds]] will be cleared.
   *       If both [[threshold]] and [[thresholds]] are set in a ParamMap, then they must be
   *       equivalent.
   *
   * Default is 0.5.
   * @group setParam
   */
  def setThreshold(value: Double): this.type = {
    if (isSet(thresholds)) clear(thresholds)
    set(threshold, value)
  }

  /**
   * Get threshold for binary classification.
   *
   * If [[threshold]] is set, returns that value.
   * Otherwise, if [[thresholds]] is set with length 2 (i.e., binary classification),
   * this returns the equivalent threshold: {{{1 / (1 + thresholds(0) / thresholds(1))}}}.
   * Otherwise, returns [[threshold]] default value.
   *
   * @group getParam
   * @throws IllegalArgumentException if [[thresholds]] is set to an array of length other than 2.
   */
  override def getThreshold: Double = {
    checkThresholdConsistency()
    if (isSet(thresholds)) {
      val ts = $(thresholds)
      require(ts.length == 2, "Factorization Machine getThreshold only applies to" +
        " binary classification, but thresholds has length != 2.  thresholds: " + ts.mkString(","))
      1.0 / (1.0 + ts(0) / ts(1))
    } else {
      $(threshold)
    }
  }

  /**
   * Set thresholds in multiclass (or binary) classification to adjust the probability of
   * predicting each class. Array must have length equal to the number of classes, with values >= 0.
   * The class with largest value p/t is predicted, where p is the original probability of that
   * class and t is the class' threshold.
   *
   * Note: When [[setThresholds()]] is called, any user-set value for [[threshold]] will be cleared.
   *       If both [[threshold]] and [[thresholds]] are set in a ParamMap, then they must be
   *       equivalent.
   *
   * @group setParam
   */
  def setThresholds(value: Array[Double]): this.type = {
    if (isSet(threshold)) clear(threshold)
    set(thresholds, value)
  }

  /**
   * Get thresholds for binary or multiclass classification.
   *
   * If [[thresholds]] is set, return its value.
   * Otherwise, if [[threshold]] is set, return the equivalent thresholds for binary
   * classification: (1-threshold, threshold).
   * If neither are set, throw an exception.
   *
   * @group getParam
   */
  override def getThresholds: Array[Double] = {
    checkThresholdConsistency()
    if (!isSet(thresholds) && isSet(threshold)) {
      val t = $(threshold)
      Array(1-t, t)
    } else {
      $(thresholds)
    }
  }

  /**
   * If [[threshold]] and [[thresholds]] are both set, ensures they are consistent.
   * @throws IllegalArgumentException if [[threshold]] and [[thresholds]] are not equivalent
   */
  protected def checkThresholdConsistency(): Unit = {
    if (isSet(threshold) && isSet(thresholds)) {
      val ts = $(thresholds)
      require(ts.length == 2, "Factorization Machine found inconsistent values for threshold and" +
        s" thresholds.  Param threshold is set (${$(threshold)}), indicating binary" +
        s" classification, but Param thresholds is set with length ${ts.length}." +
        " Clear one Param value to fix this problem.")
      val t = 1.0 / (1.0 + ts(0) / ts(1))
      require(math.abs($(threshold) - t) < 1E-5, "Factorization Machine getThreshold found" +
        s" inconsistent values for threshold (${$(threshold)}) and thresholds (equivalent to $t)")
    }
  }

  override def validateParams(): Unit = {
    checkThresholdConsistency()
  }
}


/**
 * :: Experimental ::
 * Factorization Machine.
 * Currently, this class only supports binary classification.  It will support multiclass
 * in the future.
 */
@Experimental
class FactorizationMachine(override val uid: String,
                           val latentDimension: Int)
  extends ProbabilisticClassifier[Vector, FactorizationMachine, FactorizationMachineModel]
  with FactorizationMachineParams with Logging {

  require(latentDimension >= 0,
    "Latent dimensionality must be a non-negative whole number. " +
      s"Found this.latentDimension = $latentDimension.")


  def this(latentDimension: Int) = this(
    Identifiable.randomUID("factormachine"),
    latentDimension = latentDimension)

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   * @group setParam
   */
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
   * For 0 < alpha < 1, the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   * @group setParam
   */
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
   * Whether to fit an intercept term.
   * Default is true.
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users. Note that with/without standardization,
   * the models should be always converged to the same solution when no regularization
   * is applied. In R's GLMNET package, the default behavior is true as well.
   * Default is true.
   * @group setParam
   */
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  override def setThreshold(value: Double): this.type = super.setThreshold(value)

  override def getThreshold: Double = super.getThreshold

  /**
   * Whether to over-/under-sample training instances according to the given weights in weightCol.
   * If empty, all instances are treated equally (weight 1.0).
   * Default is empty, so all instances have weight one.
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  override def setThresholds(value: Array[Double]): this.type = super.setThresholds(value)

  override def getThresholds: Array[Double] = super.getThresholds

  override protected def train(dataset: DataFrame): FactorizationMachineModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
    }

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val (summarizer, labelSummarizer) = {
      val seqOp = (c: (MultivariateOnlineSummarizer, NewMultiClassSummarizer),
                   instance: Instance) =>
        (c._1.add(instance.features, instance.weight), c._2.add(instance.label, instance.weight))

      val combOp = (c1: (MultivariateOnlineSummarizer, NewMultiClassSummarizer),
                    c2: (MultivariateOnlineSummarizer, NewMultiClassSummarizer)) =>
        (c1._1.merge(c2._1), c1._2.merge(c2._2))

      instances.treeAggregate(
        new MultivariateOnlineSummarizer, new NewMultiClassSummarizer)(seqOp, combOp)
    }

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length
    val numFeatures = summarizer.mean.size

    if (numInvalid != 0) {
      val msg = s"Classification labels should be in {0 to ${numClasses - 1} " +
        s"Found $numInvalid invalid labels."
      logError(msg)
      throw new SparkException(msg)
    }

    if (numClasses > 2) {
      val msg = s"Currently, FactorizationMachine with ElasticNet in ML package only supports " +
        s"binary classification. Found $numClasses in the input dataset."
      logError(msg)
      throw new SparkException(msg)
    }
    val featuresMean = summarizer.mean.toArray
    val featuresStd = summarizer.variance.toArray.map(math.sqrt)

    val regParamL1 = $(elasticNetParam) * $(regParam)
    val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

    val costFun = new FMCostFun(
      instances,
      numClasses,
      latentDimension,
      $(fitIntercept),
      $(standardization),
      featuresStd,
      featuresMean,
      regParamL2)

    val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
      new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    } else {
      def regParamL1Fun = (index: Int) => {
        // Remove the L1 penalization on the intercept
        if (index == 0) {
          0.0
        } else {
          if ($(standardization)) {
            regParamL1
          } else {
            // If `standardization` is false, we still standardize the data
            // to improve the rate of convergence; as a result, we have to
            // perform this reverse standardization by penalizing each component
            // differently to get effectively the same objective function when
            // the training dataset is not standardized.
            if (featuresStd(index) != 0.0) regParamL1 / featuresStd(index) else 0.0
          }
        }
      }
      new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, regParamL1Fun, $(tol))
    }

    /*
     * Initialize factorization Machine weights
     *
     * Note: Since, gradients for weights of quadratic terms are
     * proportional to the current values, for learning to be possible
     * one needs to set them to non-zero random values
     */
    val initialCoefficients: FMWeights = new FMWeights(
      intercept =
        if ($(fitIntercept)) math.log(histogram(1) / histogram(0) ) else 0.0,
      linear =
        Vectors.zeros(numFeatures),
      quadratic = generateScaledNormalRandomMatrix(
        numFeatures, latentDimension, scale = 1e-2)
    )

    val initCoeffsBreezeVector = initialCoefficients.toBreezeVector

    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initCoeffsBreezeVector)


    val (intercept, linearCoefficients, quadraticCoefficients, objectiveHistory) = {
      /*
         Note that in Logistic Regression, the objective history (loss + regularization)
         is log-likelihood which is invariance under feature standardization. As a result,
         the objective history from optimizer is the same as the one in the original space.
       */
      val arrayBuilder = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null
      while (states.hasNext) {
        state = states.next()
        arrayBuilder += state.adjustedValue
      }

      if (state == null) {
        val msg = s"${optimizer.getClass.getName} failed."
        logError(msg)
        throw new SparkException(msg)
      }

      /*
         The coefficients are trained in the scaled space; we're converting them back to
         the original space.
         Note that the intercept in scaled space and original space is the same;
         as a result, no scaling is needed.
       */
      val rawWeights = FMWeights.fromBreeze(state.x, numFeatures, latentDimension)
      val rawLinearCoefficientsArray = rawWeights.linear.toArray.clone()
      var rawQuadraticCoeffs = rawWeights.quadratic
      var i = 0
      while (i < numFeatures) {
        rawLinearCoefficientsArray(i) *= { if (featuresStd(i) != 0.0) 1.0 / featuresStd(i) else 0.0 }
        i += 1
      }

      if ($(fitIntercept)) {
        (rawWeights.intercept, Vectors.dense(rawLinearCoefficientsArray).compressed, rawQuadraticCoeffs,
          arrayBuilder.result())
      } else {
        (0.0, Vectors.dense(rawLinearCoefficientsArray).compressed, rawQuadraticCoeffs, arrayBuilder.result())
      }
    }

    if (handlePersistence) instances.unpersist()

    val model = copyValues(
      new FactorizationMachineModel(uid,
        new FMWeights(
          intercept = intercept,
          linear = linearCoefficients,
          quadratic = quadraticCoefficients
        )
      )
    )
    val fmachineSummary = new BinaryFactorizationMachineTrainingSummary(
      model.transform(dataset),
      $(probabilityCol),
      $(labelCol),
      objectiveHistory)
    model.setSummary(fmachineSummary)
  }

  override def copy(extra: ParamMap): FactorizationMachine = defaultCopy(extra)


  /**
   * Utility function for generating a "scaled" random DenseMatrix
   * @param numRows number of rows in the resulting Matrix
   * @param numCols number of columns in the resulting Matrix
   * @param scale the value/sigma with which the 'expanse' of the normal distribution to be scaled
   *              Note: the resulting distribution is no longer normalized
   * @return a DenseMatrix with values distributed as scale * normal distribution
   */

  private def generateScaledNormalRandomMatrix(numRows: Int,
                                               numCols: Int,
                                               scale: Double): DenseMatrix = {
    // Random generator
    val rng = new Random(seed = System.currentTimeMillis())


    val values: Array[Double] = (0 to numRows-1).flatMap(row => {
      (0 to numCols - 1).map(col => {
        scale * rng.nextGaussian()
      })
    }).toArray

    new DenseMatrix(numRows, numCols, values)
  }

}


/**
 * :: Experimental ::
 * Model produced by [[FactorizationMachine]].
 */
@Experimental
class FactorizationMachineModel private[ml] (
                                              override val uid: String,
                                              val weights: FMWeights)
  extends ProbabilisticClassificationModel[Vector, FactorizationMachineModel]
  with FactorizationMachineParams {

  // TODO: Add override after Spark 1.6
  val numFeatures: Int = weights.linear.size

  val latentDimension: Int = weights.quadratic.numCols


  require(weights.quadratic.numRows == numFeatures,
    "Quadratic weights must be a (numberOfFeatures x latentDimension) matrix. " +
      s"Found this.numberOfFeatures = $numFeatures, " +
      s"and this.quadratic.dim = (${weights.quadratic.numRows} x ${weights.quadratic.numCols})")

  /*
   * If all elements of Quadratic weights are zero,
   * because update rule is proportional to the current value,
   * model will not learn.
   *
   * Set quadratic weights to non-zero values.
   */
  if (weights.quadratic.numNonzeros == 0)
    log.error("Model does not learn the interactions, unless initialized with non-zero quadratic weights.")



  override val numClasses: Int = 2


  override def setThresholds(value: Array[Double]): this.type = super.setThresholds(value)

  override def getThresholds: Array[Double] = super.getThresholds

  /** Margin (rawPrediction) for class label 1.  For binary classification only. */
  // TODO: NEED CHANGE
  private val margin: Vector => Double = (features) => {
    BLAS.dot(features, weights.linear) + weights.intercept
  }

  /** Score (probability) for class label 1.  For binary classification only. */
  // TODO: Perhaps needs change if more generalized FM
  private val score: Vector => Double = (features) => {
    val m = margin(features)
    1.0 / (1.0 + math.exp(-m))
  }


  private var trainingSummary: Option[FactorizationMachineTrainingSummary] = None

  /**
   * Gets summary of model on training set. An exception is
   * thrown if `trainingSummary == None`.
   */
  def summary: FactorizationMachineTrainingSummary = trainingSummary match {
    case Some(summ) => summ
    case None =>
      throw new SparkException(
        "No training summary available for this FactorizationMachineModel",
        new NullPointerException())
  }

  private[classification] def setSummary(summary: FactorizationMachineTrainingSummary)
  : this.type = {
    this.trainingSummary = Some(summary)
    this
  }

  /** Indicates whether a training summary exists for this model instance. */
  def hasSummary: Boolean = trainingSummary.isDefined

  /**
   * Evaluates the model on a testset.
   * @param dataset Test dataset to evaluate model on.
   */
  // TODO: decide on a good name before exposing to public API
  private[classification] def evaluate(dataset: DataFrame): FactorizationMachineSummary = {
    new BinaryFactorizationMachineSummary(this.transform(dataset), $(probabilityCol), $(labelCol))
  }

  /**
   * Predict label for the given feature vector.
   * The behavior of this can be adjusted using [[thresholds]].
   */
  override protected def predict(features: Vector): Double = {
    // Note: We should use getThreshold instead of $(threshold) since getThreshold is overridden.
    if (score(features) > getThreshold) 1 else 0
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        while (i < size) {
          dv.values(i) = 1.0 / (1.0 + math.exp(-dv.values(i)))
          i += 1
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in FactorizationMachineModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val m = margin(features)
    Vectors.dense(-m, m)
  }


  override def copy(extra: ParamMap): FactorizationMachineModel = {
    val newModel = copyValues(new FactorizationMachineModel(uid, weights), extra)
    if (trainingSummary.isDefined) newModel.setSummary(trainingSummary.get)
    newModel.setParent(parent)
  }

  override protected def raw2prediction(rawPrediction: Vector): Double = {
    val t = getThreshold
    val rawThreshold = if (t == 0.0) {
      Double.NegativeInfinity
    } else if (t == 1.0) {
      Double.PositiveInfinity
    } else {
      math.log(t / (1.0 - t))
    }
    if (rawPrediction(1) > rawThreshold) 1 else 0
  }

  override protected def probability2prediction(probability: Vector): Double = {
    if (probability(1) > getThreshold) 1 else 0
  }

}

/**
 * Abstraction for multinomial Factorization Machine Training results.
 * Currently, the training summary ignores the training weights except
 * for the objective trace.
 */
sealed trait FactorizationMachineTrainingSummary extends FactorizationMachineSummary {

  /** objective function (scaled loss + regularization) at each iteration. */
  def objectiveHistory: Array[Double]

  /** Number of training iterations until termination */
  def totalIterations: Int = objectiveHistory.length

}

/**
 * Abstraction for Factorization Machine Results for a given model.
 */
sealed trait FactorizationMachineSummary extends Serializable {

  /** Dataframe outputted by the model's `transform` method. */
  def predictions: DataFrame

  /** Field in "predictions" which gives the calibrated probability of each instance as a vector. */
  def probabilityCol: String

  /** Field in "predictions" which gives the the true label of each instance. */
  def labelCol: String

}

/**
 * :: Experimental ::
 * Factorization Machine training results.
 * @param predictions dataframe outputted by the model's `transform` method.
 * @param probabilityCol field in "predictions" which gives the calibrated probability of
 *                       each instance as a vector.
 * @param labelCol field in "predictions" which gives the true label of each instance.
 * @param objectiveHistory objective function (scaled loss + regularization) at each iteration.
 */
@Experimental
class BinaryFactorizationMachineTrainingSummary private[classification] (
                                                                          predictions: DataFrame,
                                                                          probabilityCol: String,
                                                                          labelCol: String,
                                                                          val objectiveHistory: Array[Double])
  extends BinaryLogisticRegressionSummary(predictions, probabilityCol, labelCol)
  with FactorizationMachineTrainingSummary {

}

/**
 * :: Experimental ::
 * Binary Logistic regression results for a given model.
 * @param predictions dataframe outputted by the model's `transform` method.
 * @param probabilityCol field in "predictions" which gives the calibrated probability of
 *                       each instance.
 * @param labelCol field in "predictions" which gives the true label of each instance.
 */
@Experimental
class BinaryFactorizationMachineSummary private[classification] (@transient override val predictions: DataFrame,
                                                                 override val probabilityCol: String,
                                                                 override val labelCol: String) extends FactorizationMachineSummary {

  private val sqlContext = predictions.sqlContext
  import sqlContext.implicits._

  /**
   * Returns a BinaryClassificationMetrics object.
   */
  // TODO: Allow the user to vary the number of bins using a setBins method in
  // BinaryClassificationMetrics. For now the default is set to 100.
  @transient private val binaryMetrics = new BinaryClassificationMetrics(
    predictions.select(probabilityCol, labelCol).map {
      case Row(score: Vector, label: Double) => (score(1), label)
    }, 100
  )

  /**
   * Returns the receiver operating characteristic (ROC) curve,
   * which is an Dataframe having two fields (FPR, TPR)
   * with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
   * @see http://en.wikipedia.org/wiki/Receiver_operating_characteristic
   */
  @transient lazy val roc: DataFrame = binaryMetrics.roc().toDF("FPR", "TPR")

  /**
   * Computes the area under the receiver operating characteristic (ROC) curve.
   */
  lazy val areaUnderROC: Double = binaryMetrics.areaUnderROC()

  /**
   * Returns the precision-recall curve, which is an Dataframe containing
   * two fields recall, precision with (0.0, 1.0) prepended to it.
   */
  @transient lazy val pr: DataFrame = binaryMetrics.pr().toDF("recall", "precision")

  /**
   * Returns a dataframe with two fields (threshold, F-Measure) curve with beta = 1.0.
   */
  @transient lazy val fMeasureByThreshold: DataFrame = {
    binaryMetrics.fMeasureByThreshold().toDF("threshold", "F-Measure")
  }

  /**
   * Returns a dataframe with two fields (threshold, precision) curve.
   * Every possible probability obtained in transforming the dataset are used
   * as thresholds used in calculating the precision.
   */
  @transient lazy val precisionByThreshold: DataFrame = {
    binaryMetrics.precisionByThreshold().toDF("threshold", "precision")
  }

  /**
   * Returns a dataframe with two fields (threshold, recall) curve.
   * Every possible probability obtained in transforming the dataset are used
   * as thresholds used in calculating the recall.
   */
  @transient lazy val recallByThreshold: DataFrame = {
    binaryMetrics.recallByThreshold().toDF("threshold", "recall")
  }
}


/**
 * FMAggregator computes the gradient and loss for binary logistic loss function, as used
 * in binary classification for instances in sparse or dense vector in a online fashion.
 *
 * Note that multinomial logistic loss is not supported yet!
 *
 * Two FMAggregator can be merged together to have a summary of loss and gradient of
 * the corresponding joint dataset.
 *
 * @param coefficients The coefficients corresponding to the features.
 * @param numClasses the number of possible outcomes for k classes classification problem in
 *                   Multinomial Logistic Regression.
 * @param fitIntercept Whether to fit an intercept term.
 * @param featuresStd The standard deviation values of the features.
 * @param featuresMean The mean values of the features.
 */
private class FMAggregator(coefficients: FMWeights,
                           numClasses: Int,
                           fitIntercept: Boolean,
                           featuresStd: Array[Double],
                           featuresMean: Array[Double]) extends Serializable {


  private val numFeatures = coefficients.linear.size
  private val latentDimension = coefficients.quadratic.numCols

  private var weightSum = 0.0       // no change
  private var lossSum = 0.0         // no change either
  private var gradientSum = new FMWeights(
      intercept = 0.0,
      linear = Vectors.zeros(numFeatures),
      quadratic = DenseMatrix.zeros(numFeatures, coefficients.quadratic.numCols)
    )

  /**
   * Add a new training instance to this FMAggregator, and update the loss and gradient
   * of the objective function.
   *
   * @param instance The instance of data point to be added.
   * @return This FMAggregator object.
   */
  def add(instance: Instance): this.type = {
    instance match { case Instance(label, weight, features) =>
      require(features.size == numFeatures, s"Dimensions mismatch when adding new instance." +
        s" Expecting $numFeatures but got ${features.size}.")
      require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

      if (weight == 0.0) return this

      var currentInterceptGradient = 0.0
      val currentLinearGradientValues = Array.ofDim[Double](numFeatures)
      numClasses match {
        case 2 =>
          // For Binary.
          val margin = - {
            var sum = 0.0
            features.foreachActive { (index, value) =>
              if (featuresStd(index) != 0.0 && value != 0.0) {
                sum += coefficients.linear(index) * (value / featuresStd(index))
              }
            }
            sum + {
              if (fitIntercept) coefficients.intercept else 0.0
            }
          }

          val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - label)

          features.foreachActive { (index, value) =>
            if (featuresStd(index) != 0.0 && value != 0.0) {
              currentLinearGradientValues(index) = multiplier * (value / featuresStd(index))
            }
          }

          if (fitIntercept) {
            currentInterceptGradient = multiplier
          }

          if (label > 0) {
            // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
            lossSum += weight * MLUtils.log1pExp(margin)
          } else {
            lossSum += weight * (MLUtils.log1pExp(margin) - margin)
          }
        case _ =>
          new NotImplementedError("FactorizationMachine with ElasticNet in ML package " +
            "only supports binary classification for now.")
      }
      weightSum += weight
      gradientSum += new FMWeights(currentInterceptGradient,
        Vectors.dense(currentLinearGradientValues),
        DenseMatrix.zeros(numFeatures, gradientSum.quadratic.numCols))                   //TODO: NEED CHANGE
      this
    }
  }

  /**
   * Merge another FMAggregator, and update the loss and gradient
   * of the objective function.
   * (Note that it's in place merging; as a result, `this` object will be modified.)
   *
   * @param other The other LogisticAggregator to be merged.
   * @return This FMAggregator object.
   */
  def merge(other: FMAggregator): this.type = {
    require(numFeatures == other.numFeatures && latentDimension == other.latentDimension,
      s"Dimensions mismatch when merging with another " +
        s"LeastSquaresAggregator. Expecting coefficients with " +
        s"numFeatures = $numFeatures and latentDimension = $latentDimension, " +
        s"but got ${other.numFeatures} and ${other.latentDimension}.")

    if (other.weightSum != 0.0) {
      weightSum += other.weightSum
      lossSum += other.lossSum
      gradientSum += other.gradientSum
    }
    this
  }

  def loss: Double = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    lossSum / weightSum
  }

  def gradient: FMWeights = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    gradientSum / weightSum
  }
}

/**
 * FMCostFun implements Breeze's DiffFunction[T] for a multinomial logistic loss function,
 * as used in multi-class classification (it is also used in binary logistic regression).
 * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
 * It's used in Breeze's convex optimization routines.
 */
private class FMCostFun(instances: RDD[Instance],
                        numClasses: Int,
                        latentDimension: Int,
                        fitIntercept: Boolean,
                        standardization: Boolean,
                        featuresStd: Array[Double],
                        featuresMean: Array[Double],
                        regParamL2: Double) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val numFeatures = featuresStd.length
    val coeffs:FMWeights = FMWeights.fromBreeze(coefficients, numFeatures, latentDimension)

    val fmAggregator = {
      val seqOp = (c: FMAggregator, instance: Instance) => c.add(instance)
      val combOp = (c1: FMAggregator, c2: FMAggregator) => c1.merge(c2)

      instances.treeAggregate(
        new FMAggregator(coeffs, numClasses, fitIntercept, featuresStd, featuresMean)
      )(seqOp, combOp)
    }

    /*
     * Regularization
     *
     * For intercept, no regularization
     */
    val regularizationLinear = Array.ofDim[Double](numFeatures)

    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParamL2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.linear.foreachActive { (index, value) =>
        // If `fitIntercept` is true, the last term which is intercept doesn't
        // contribute to the regularization.
        if (index != numFeatures) {
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            if (standardization) {
              regularizationLinear(index) += regParamL2 * value
              value * value
            } else {
              if (featuresStd(index) != 0.0) {
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                val temp = value / (featuresStd(index) * featuresStd(index))
                regularizationLinear(index) += regParamL2 * temp
                value * temp
              } else {
                0.0
              }
            }
          }
        }
      }
      0.5 * regParamL2 * sum
    }

    val regGradient  = new FMWeights(
      intercept = 0.0,
      linear = Vectors.dense(regularizationLinear),
      quadratic = DenseMatrix.zeros(numFeatures, coeffs.quadratic.numCols)                //TODO: NEED CHANGE
    )

    (fmAggregator.loss + regVal, (fmAggregator.gradient + regGradient).toBreezeVector)
  }
}




/**
 * :: Experimental ::
 * Model weights for [[FactorizationMachineModel]].
 */
@Experimental
private[classification] class FMWeights(val intercept: Double,
                                        val linear: Vector,
                                        val quadratic: DenseMatrix) extends Serializable {

  /*
   * Number of features
   */
  val numFeatures = linear.size

  /*
   * Latent dimensionality
   */
  val latentDimension = quadratic.numCols


  /*
   * Quadratice weight Matrix should be numFeatures x k
   */
  require(quadratic.numRows == numFeatures,
    "The quadratic weights should have the same number of features as in linear component. " +
      s"Found, this.linear.size = $numFeatures, " +
      s"and this.quadratic.numRows = ${quadratic.numRows}")


  def +(other: FMWeights): FMWeights = {

    require(other.linear.size == numFeatures,
      "The linear weights sizes do not match. " +
        "Found, this.linear.size = " + numFeatures +
        " other.linear.size = " + other.linear.size)

    require(other.quadratic.numRows == numFeatures &&
      other.quadratic.numCols == latentDimension,
      "The Quadratic weights sizes do not match. " +
        "Found, this.quadratic = (" + numFeatures + ", " + latentDimension + ")" +
        " other.quadratic = (" + other.quadratic.numRows + ", " + other.quadratic.numCols + ")")


    val outputLinear = Vectors.fromBreeze(
      linear.toBreeze + other.linear.toBreeze
    )
    val outputQuadratic = quadratic
    new F2jBLAS().daxpy(numFeatures * latentDimension, 1.0,
      other.quadratic.values, 1, outputQuadratic.values, 1)

    new FMWeights(
      intercept + other.intercept,
      outputLinear,
      outputQuadratic
    )
  }

  def *(scale: Double): FMWeights = {

    if (scale == 0)
      new FMWeights(0.0,
        Vectors.zeros(numFeatures),
        DenseMatrix.zeros(numFeatures, latentDimension))

    else {

      /*
     * Linear components
     */
      val scaledLinear = Vectors.fromBreeze(scale * linear.toBreeze)

      /*
     * Quadratic component
     */
      val scaledQuadratic =
        new DenseMatrix(quadratic.numRows, quadratic.numCols, quadratic.toArray.map(value => scale * value))

      new FMWeights(
        scale * intercept,
        scaledLinear,
        scaledQuadratic
      )
    }
  }

  def -(other: FMWeights) = this + (other * (-1.0))


  def /(scale: Double): FMWeights = {

    require(scale != 0.0,
      "division by zero")

    this * (1.0 / scale)

  }

}

private[classification] object FMWeights {

  implicit class scaleFmw(val scale: Double) {
    def *(fMWeights: FMWeights) = fMWeights * scale
  }

  implicit class FmwUtils(val fMWeights: FMWeights) {
    /**
     * Convert an FMWeights object to Breeze Matrix
     * (input implicit)
     * @return a Breeze vector
     *         of length 1 + numFeatures + numFeatures x latentDimension
     *          => element 0 = intercept
     *          => elements 1 to numFeatures = linear weights
     *          => remaining are quadratic weights
     */
    def toBreezeVector: BDV[Double] = {

      val numFeatures = fMWeights.numFeatures
      val latentDimension = fMWeights.latentDimension

      val outputBdv: BDV[Double] = BDV.zeros[Double](1 + numFeatures + numFeatures * latentDimension)

      // Column offset
      var runner: Int = 0

      // Intecept is the only non-zero component in the fist column
      outputBdv(runner) = fMWeights.intercept
      runner += 1

      //Linear weights as second column
      for (idx <- 0 to numFeatures - 1) {
        outputBdv(runner) = fMWeights.linear(idx)
        runner += 1
      }

      // Quadratic weights go as the columns from third onwards
      for (row <- 0 to numFeatures - 1)
        for (col <- 0 to latentDimension - 1) {
          outputBdv(runner) = fMWeights.quadratic(row, col)
          runner += 1
        }

      outputBdv
    }
  }

  def fromBreeze(bdv: BDV[Double],
                 numFeatures: Int,
                 latentDimension: Int): FMWeights = {

    // unwrap the Breeze vector to FMWeights
    var runner: Int = 0

    // intercept
    val intercept = bdv(runner)
    runner += 1

    // Linear weights are second column
    val linear = Vectors.fromBreeze(bdv(runner to runner + numFeatures - 1))
    runner += numFeatures

    // Quadratic weights are everything from third column onwards
    val quadratic = new DenseMatrix(numFeatures, latentDimension,
      bdv(runner to bdv.length - 1).toArray, isTransposed = true)

    new FMWeights(intercept, linear, quadratic)
  }

}
