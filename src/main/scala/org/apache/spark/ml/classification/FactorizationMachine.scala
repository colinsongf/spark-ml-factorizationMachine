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

import breeze.optimize.AdaptiveGradientDescent.L2Regularization
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}

import scala.collection.mutable
import scala.util.Random

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS, OWLQN}

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

/**
 * Params for Factorization Machine.
 */
private[classification] trait FactorizationMachineParams extends PredictorParams
with HasRegParam with HasElasticNetParam with HasMaxIter with HasFitIntercept with HasTol
with HasStandardization with HasWeightCol {
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
  extends Predictor[Vector, FactorizationMachine, FactorizationMachineModel]
  with FactorizationMachineParams
  with Logging {

  def this(latentDimension: Int) = this(
    Identifiable.randomUID("factormachine"),
    latentDimension = latentDimension)


  require(latentDimension >= 0,
    "Latent dimensionality must be a non-negative whole number. " +
      s"Found this.latentDimension = $latentDimension.")

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


  /**
   * Whether to over-/under-sample training instances according to the given weights in weightCol.
   * If empty, all instances are treated equally (weight 1.0).
   * Default is empty, so all instances have weight one.
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")


  override protected def train(dataset: DataFrame): FactorizationMachineModel = {
    // Extract data
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
    }

    // Data persistence: persist only if unpersisted before
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
      val msg = s"Currently, FactorizationMachine with ElasticNet in ML package supports " +
        s"only binary classification. Found $numClasses in the input dataset."
      logError(msg)
      throw new SparkException(msg)
    }
    val featuresMean = summarizer.mean.toArray
    val featuresStd = summarizer.variance.toArray.map(math.sqrt)

    val costFun = new FMCostFun(
      instances,
      numClasses,
      latentDimension,
      $(fitIntercept),
      $(standardization),
      featuresStd,
      featuresMean,
      $(regParam),
      $(elasticNetParam))

    val optimizer = new L2Regularization[BDV[Double]](
      $(regParam),
      maxIter = $(maxIter), stepSize = 4, tolerance = $(tol))

    /*
     * Initialize factorization Machine weights
     *
     * Note: Since, gradients for weights of quadratic terms are
     * proportional to the current values, for learning to be possible
     * one needs to set them to non-zero random values
     */
    val initialCoefficients: FMCoefficients = new FMCoefficients(
      intercept =
        if ($(fitIntercept)) math.log( histogram(1) / histogram(0) ) else 0.0,
      linear =
        Vectors.zeros(numFeatures),
      quadratic = generateScaledNormalRandomMatrix(
        numFeatures, latentDimension, scale = 1e-2)
    )

    // All the coefficients laid as a long vector
    val initCoefficientLongVector = initialCoefficients.flattenToBreezeVector

    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initCoefficientLongVector)


    val (coefficients, objectiveHistory) = {
      /*
         Note that in Factorization Machine, the objective history (loss + regularization)
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
      val rawCoefficients = state.x.copy
      var runner = 1   // no scaling for intercept
      while (runner < rawCoefficients.length) {
        // linear weights
        if (runner <= numFeatures)
          rawCoefficients(runner) *= {
            if (featuresStd(runner - 1) != 0.0)
              1.0 / featuresStd(runner - 1)
            else 0.0
          }
        else {
          // quadratic weights
          val (row, _) = FMCoefficients.getRowColIndexFromPosition(runner, numFeatures, latentDimension)
          rawCoefficients(runner) *= {
            if (featuresStd(row) != 0.0)
              1.0 / featuresStd(row)
            else 0.0
          }
        }

        runner += 1
      }

      if ( !$(fitIntercept) )
        rawCoefficients(0) = 0.0


      ( FMCoefficients.fromBreezeVector(rawCoefficients, numFeatures, latentDimension),
        arrayBuilder.result()
        )
    }

    if (handlePersistence) instances.unpersist()


    val model = copyValues(
      new FactorizationMachineModel(uid, coefficients)
    )

    model
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
//TODO: make it private again
object FactorizationMachineModel {

  case class MarginData(margin: Double, auxiliaryVector: BDV[Double])

  def MarginAndAuxiliarySum(features: Vector,
                            coefficients: FMCoefficients,
                            withIntercept: Boolean = true,
                            featuresStd: Option[Array[Double]] = None)
  : MarginData = {

    // Standardized features
    val standardizedFeatures = featuresStd match {
      case Some(f) => Vectors.sparse(features.size,
        features.toSparse.indices
          .filter(idx => f(idx) != 0.0)
          .map(idx => (idx, features(idx) / f(idx))).toSeq
      )
      case None => features
    }


    // Active (sparse) features
    val activeFeatures = standardizedFeatures.toSparse.indices


    // Auxiliary Vector
    // k'th component = sum_j v_jk * feat_j
    // Used (again) for calculating gradient w. r. t. quadratic weights
    val auxiliaryVector = BDV.zeros[Double](coefficients.latentDimension)

    var margin = 0.0

    // intercept
    margin += {
      if (withIntercept) coefficients.intercept
      else 0.0
    }

    // linear
    margin +=
      coefficients.linear.toBreeze.dot(standardizedFeatures.toBreeze)

    // quadratic
    margin +=       // TODO: NEED CHECK OF LOGIC
      (0 to coefficients.quadratic.numCols - 1).map{col =>

        // Sum over j
        // The individual terms: v_jk * feat_j
        // and the corresponding squares: v_jk^2 * feat_j^2
        val termsAndSquaredTerms = activeFeatures
          .map(idx => coefficients.quadratic(idx, col) * standardizedFeatures(idx)   )   // v_idx,col * f_idx
          .map(value => List(value, math.pow(value, 2)) )    // (v_ij * f_i, v_ij^2f_i^2)
          .toList              // (v1, v1^2) (v2, v2^2) ....
          .transpose           // (v1,v2, v3 ...), (v1^2, v2^2, ....)

        // Sum over k
        // Sum of v_jk * feat_j
        // and sum of v_jk^2 * feat_j^2
        val sums = termsAndSquaredTerms
          .map(_.sum)            // (v1 + v2 + ....), (v1^2 + v2^2 + ...)

        // Only for clarity
        val (sumOfTerms, sumOfSquaredTerms) = (sums.head, sums.last)

        // j'th component of auxVec = sum_i v_ij * f_i
        auxiliaryVector(col) = sumOfTerms

        0.5 * (math.pow(sumOfTerms, 2) - sumOfSquaredTerms)
      }.sum

    MarginData(margin, auxiliaryVector)
  }
}



@Experimental
class FactorizationMachineModel private[ml](override val uid: String,
                                            val coefficients: FMCoefficients)
  extends PredictionModel[Vector, FactorizationMachineModel]
  with FactorizationMachineParams {


  import FactorizationMachineModel._


  // TODO: Add override after Spark v > 1.5.1
  val numFeatures: Int = coefficients.linear.size

  val latentDimension: Int = coefficients.quadratic.numCols

  require(coefficients.quadratic.numRows == numFeatures,
    "Number of rows in quadratic weight matrix must be equal to numFeatures " +
      "(derived from linear weights size). " +
      s"Found this.numberOfFeatures = $numFeatures, " +
      s"and this.quadratic.numRows = ${coefficients.quadratic.numRows}.")

  /*
   * If all elements of Quadratic weights are zero,
   * because update rule is proportional to the current value,
   * model will not learn interactions.
   */
  if (latentDimension > 0 && coefficients.quadratic.numNonzeros == 0)
    logWarning("ModelInitializationWarning: " +
      "Model will not learn the interactions further, " +
      "unless initialized with non-zero quadratic weights.")


  val numClasses: Int = 2

  /** Score (probability) for class label 1.  For binary classification only. */
  // TODO: Perhaps needs change if more generalized FM
  private val score: Vector => Double = (features) => {
    val m = MarginAndAuxiliarySum(features, coefficients).margin
    math.signum(m)
  }

  /**
   * Predict label for the given feature vector.
   */
  override protected def predict(features: Vector): Double = score(features)


  override def copy(extra: ParamMap): FactorizationMachineModel = {
    val newModel = copyValues(new FactorizationMachineModel(uid, coefficients), extra)
    newModel.setParent(parent)
  }

}


/**
 * FMAggregator computes the gradient and loss for binary Factorization Machine loss function,
 * as used in binary classification for instances in sparse or dense vector in a online fashion.
 *
 * Note that multinomial classification is not supported yet!
 *
 * Two FMAggregator can be merged together to have a summary of loss and gradient of
 * the corresponding joint dataset.
 *
 * @param coefficients The coefficients corresponding to the features.
 * @param numClasses the number of possible outcomes for k classes classification problem in
 *                   Multinomial Factorization machine.
 * @param fitIntercept Whether to fit an intercept term.
 * @param featuresStd The standard deviation values of the features.
 * @param featuresMean The mean values of the features.
 */
private class FMAggregator(coefficients: FMCoefficients,
                           numClasses: Int,
                           fitIntercept: Boolean,
                           featuresStd: Array[Double],
                           featuresMean: Array[Double]) extends Serializable {

  import FactorizationMachineModel._

  private val numFeatures = coefficients.linear.size
  private val latentDimension = coefficients.quadratic.numCols

  private var weightSum = 0.0       // no change
  private var lossSum = 0.0         // no change either
  private var gradientSum = new FMCoefficients(
      intercept = 0.0,
      linear = Vectors.zeros(numFeatures),
      quadratic = DenseMatrix.zeros(numFeatures, latentDimension)
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

      val currentGradientVector: BDV[Double] =
        BDV.zeros[Double](1 + numFeatures + numFeatures * latentDimension)

      numClasses match {
        case 2 =>
          // For Binary.
          val marginAndAuxVec = MarginAndAuxiliarySum(
            features = features,
            coefficients = coefficients,
            withIntercept = fitIntercept,
            featuresStd = Some(featuresStd)
          )

          val margin = - marginAndAuxVec.margin
          val auxiliaryVector = marginAndAuxVec.auxiliaryVector

          // The output error (label - prediction)
          // Specific to Logistic Regression (and some link-loss match)
          val outputDiscrepancy = weight * (1.0 / (1.0 + math.exp(margin)) - label)

          // Gradient w.r.t. intercept = discrepancy * 1.0
          if (fitIntercept) {
            currentGradientVector(0) = outputDiscrepancy
          }

          // Gradient w.r.t. linear weight (i) =
          // discrepancy * feature(i) value
          features.foreachActive { (index, value) =>
            if (featuresStd(index) != 0.0 && value != 0.0) {
              currentGradientVector(index + 1) = outputDiscrepancy * (value / featuresStd(index))
            }
          }

          // Gradient w.r.t. quadratic weight(j, k) =
          // discrepancy * [ feat_j * auxVec_k  - w_jk * feat_j^2 ]   // TODO: Check logic
          for(index <- numFeatures + 1 to currentGradientVector.length - 1) {
            val (row, col) = FMCoefficients.getRowColIndexFromPosition(index, numFeatures, latentDimension)
            if (featuresStd(row) != 0.0 && features(row) != 0.0) {
              val value = features(row) / featuresStd(row)
              currentGradientVector(index) = outputDiscrepancy * (
                value * (
                  auxiliaryVector(col) - coefficients.quadratic(row, col) * value
                  )
                )
            }
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
      gradientSum += FMCoefficients
        .fromBreezeVector(
          currentGradientVector,
          numFeatures, latentDimension)

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

  def gradient: FMCoefficients = {
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
                        regParam: Double,
                        elasticNetParam: Double) extends DiffFunction[BDV[Double]] {


  val regParamL1 = elasticNetParam * regParam
  val regParamL2 = (1.0 - elasticNetParam) * regParam


  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val numFeatures = featuresStd.length
    val coeffs:FMCoefficients = FMCoefficients.fromBreezeVector(coefficients, numFeatures, latentDimension)

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
    val regGradientVector:BDV[Double] = BDV.zeros[Double](1 + numFeatures + numFeatures * latentDimension)

    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParam == 0.0) {
      0.0
    } else {
      var sum = 0.0
      // Contributions from linear weights
      coeffs.linear.foreachActive { (index, value) =>
        sum += {
          if (standardization) {
            regGradientVector(index + 1) = regParamL1 + regParamL2 * value
            value * value
          } else {
            if (featuresStd(index) != 0.0) {
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              val temp = value / (featuresStd(index) * featuresStd(index))
              regGradientVector(index + 1) = regParamL1 / featuresStd(index - 1) + regParamL2 * temp
              value * temp
            } else {
              0.0
            }
          }
        }
      }

      // Contributions from quadratic weights
      coeffs.quadratic.foreachActive{ (row, col, value) =>
        sum += {
          if (standardization) {
            regGradientVector(1 + numFeatures + latentDimension * row + col) = regParamL1 + regParamL2 * value
            value * value
          }
          else {
            if (featuresStd(row) != 0.0) {
              val temp = value / ( featuresStd(row) * featuresStd(row) )
              regGradientVector(1 + numFeatures + latentDimension * row + col) = regParamL1 / featuresStd(row) + regParamL2 * temp
              value * temp
            }
            else 0.0
          }
        }
      }

      0.5 * regParamL2 * sum
    }

    val regGradient  = FMCoefficients.fromBreezeVector(regGradientVector, numFeatures, latentDimension)

    (fmAggregator.loss + regVal, (fmAggregator.gradient + regGradient).flattenToBreezeVector)
  }
}




/**
 * :: Experimental ::
 * Model weights for [[FactorizationMachineModel]].
 */
@Experimental
//TODO: Make it private[classification]
class FMCoefficients(val intercept: Double,
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


  def +(other: FMCoefficients): FMCoefficients = {

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

    new FMCoefficients(
      intercept + other.intercept,
      outputLinear,
      outputQuadratic
    )
  }

  def *(scale: Double): FMCoefficients = {

    if (scale == 0)
      new FMCoefficients(0.0,
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

      new FMCoefficients(
        scale * intercept,
        scaledLinear,
        scaledQuadratic
      )
    }
  }

  def -(other: FMCoefficients) = this + (other * (-1.0))


  def /(scale: Double): FMCoefficients = {

    require(scale != 0.0,
      "division by zero")

    this * (1.0 / scale)

  }

}

//TODO: Make it private[classification]
object FMCoefficients {

  implicit class scaleFmw(val scale: Double) {
    def *(FMCoefficients: FMCoefficients) = FMCoefficients * scale
  }

  implicit class FmwUtils(val FMCoefficients: FMCoefficients) {
    /**
     * Convert an FMCoefficients object to Breeze Matrix
     * (input implicit)
     * @return a Breeze vector
     *         of length 1 + numFeatures + numFeatures x latentDimension
     *          => element 0 = intercept
     *          => elements 1 to numFeatures = linear weights
     *          => remaining are quadratic weights
     */
    def flattenToBreezeVector: BDV[Double] = {

      val numFeatures = FMCoefficients.numFeatures
      val latentDimension = FMCoefficients.latentDimension

      val outputBdv: BDV[Double] = BDV.zeros[Double](1 + numFeatures + numFeatures * latentDimension)

      // Runner along the long vector
      var runner: Int = 0

      // intercept
      outputBdv(runner) = FMCoefficients.intercept
      runner += 1

      // Linear weights
      for (idx <- 0 to numFeatures - 1) {
        outputBdv(runner) = FMCoefficients.linear(idx)
        runner += 1
      }

      // Quadratic weights
      for (row <- 0 to numFeatures - 1)
        for (col <- 0 to latentDimension - 1) {
          outputBdv(runner) = FMCoefficients.quadratic(row, col)
          runner += 1
        }

      outputBdv
    }
  }

  def fromBreezeVector(bdv: BDV[Double],
                       numFeatures: Int,
                       latentDimension: Int): FMCoefficients = {

    // unwrap the Breeze vector to FMCoefficients
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

    new FMCoefficients(intercept, linear, quadratic)
  }


  def getRowColIndexFromPosition(position: Int,
                                 numFeatures: Int,
                                 latentDimension: Int)
  : (Int, Int) = {

    require(position > numFeatures,
      s"Position $position does not correspond to any quadratic weight, " +
        s"in a FMCoefficients object of numFeatures = $numFeatures " +
        s"and latentDimension = $latentDimension")

    val adjustedIndex = position - (1 + numFeatures)
    (adjustedIndex / latentDimension,
      adjustedIndex % latentDimension)
  }

}
