/**
 * Project          : aol-ml
 * File             : Driver
 * Author           : Nikhil J Joshi (nikhil.joshi@teamaol.com)
 * Date Created     : 10/26/15
 *
 * Descriptor       : 
 * " The class usage goes here. "
 *
 */

package com.aol.advertising.ml

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.tuning.{TrainValidationSplit, CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import scala.collection.immutable.HashSet

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}

import org.apache.spark.ml.classification._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}

// Scopt Input parser case class
case class ScoptConfig(locally: Boolean = false,
                       input: String = "",
                       output: String = "",
                       model:Int = 0
                        )


object Driver {

  def composeFilesList(path: String,
                       fileTypeExtension: String)
  : Array[String] =
    if (path.endsWith(fileTypeExtension)) // A single specific file provided
      Array(path)
    else {

      val fs = FileSystem.get(new Configuration())

      val fileStatus = fs.globStatus(new Path(path))
      if (fileStatus != null) {
        fileStatus.map(f => {
          f.getPath.toString
        }).filter(_.endsWith(fileTypeExtension))
      }
      else
        throw new RuntimeException("No input file of type '" +
          fileTypeExtension + "' found at the location specified by '" +
          path)
    }


  def main(args: Array[String]): Unit = {

    /*
     *  The commandline argument handler
     */
    val parser = new scopt.OptionParser[ScoptConfig]("(sbt) run") {
      head("Commandline Arguments", "0.x")

      opt[Unit]('l', "runlocally")
        .action { (_, c) => c.copy(locally = true) }
        .text("run locally instead of on Spark cluster/client mode")

      opt[String]('i', "input")
        .required()
        .action { (x, c) => c.copy(input = x) }
        .text("input (file path) is a required property [type: String]")

      opt[String]('o', "output")
        .required()
        .action { (x, c) => c.copy(output = x) }
        .text("output (file path) is a required property [type: String]")

      opt[Int]('m', "model")
        .action{ (x, c) => c.copy(model = x)}
        .text("classification model (0: LogisticRegression, 1: FactorizationMachine)")

      help("help")
        .text("prints this usage text")
    }
    // Parse the input arguments
    val clArgs = parser.parse(args, ScoptConfig()) match {
      case Some(conf) => conf
      case _ => throw new RuntimeException("Error unrolling commandline arguments. Run 'sbt run --help'.")
    }


    /*
     * Compose input file list to be processed
     */
    val files: Array[String] = composeFilesList(clArgs.input, ".gz")

    if (files.length < 1)
      throw new RuntimeException("No file was found.")

    if (files.length > 1)
      throw new RuntimeException("Currently only one file at a time is served.")


    val trainingDataFile =
      if ( MLUtil.FileLoader.isFound(files.head) )
        files.head
      else
        throw new RuntimeException("Datafile '" + files.head + "' not found.")

    /*
     * Spark Context
     */
    val sparkConf = new SparkConf()
      .setAppName("FactorizationMachine-demo")
    if (clArgs.locally)
      sparkConf.setMaster("local[4]")

    val sc = new SparkContext(sparkConf)


    val sqlContext = SQLContext.getOrCreate(sc)
    import sqlContext.implicits._

    import MLUtil.FileLoader._



    /*
     * Features set
     */
    val columnsToSelect = HashSet(
      "isclk",
      "ad_ctr",
      "advertiser_ctr",
      "campaign_ctr",
      "site_ad_ctr",
      "site_campaign_ctr",
      "site_ctr",
      "adid",
      "browser_5",
      "cc",
      "user_cookie_ability_1604",
      "user_acid_age_10",
      "user_network_clicker_546",
      "user_network_actor_547"
    )

    /*
     * Categorical features/variables
     */
    val categoricalVariables = HashSet(
      "adid",
      "browser_5",
      "cc",
      "user_cookie_ability_1604",
      "user_acid_age_10",
      "user_network_clicker_546",
      "user_network_actor_547"
    )

    /*
     * The training data
     * (and the feature index map)
     */
    val trainingDataRddAndFeatureMap = sc.loadCsvFileAsRDD(
      trainingDataFile,
      withHeader = true,
      separator = " ",
      selectColumns = columnsToSelect,
      categoricalFeatures = categoricalVariables)

    val trainingData = trainingDataRddAndFeatureMap.data
      .repartition(sc.defaultMinPartitions * 3)
      .map(point => LabeledPoint( if (point.label == -1.0) 0.0 else point.label, point.features))
      .toDF("label", "features")



    /*
    * Test/validation data
    * (with the same loader specifications
    * as used in importing training data.)
    */
    val testData = sc.loadCsvFileAsRDD(
      trainingDataFile.replaceFirst(".TR.gz", ".VA.gz"),
      withFeatureIndexing = trainingDataRddAndFeatureMap.featureIndexMap,
      withLoaderParams = trainingDataRddAndFeatureMap.loaderParams
    ).data
      .repartition(sc.defaultMinPartitions * 3)
      .map(point => LabeledPoint( if (point.label == -1.0) 0.0 else point.label, point.features))
      .toDF("label", "features")


    /*
     * Logistic Regression or Factorization Machine model
     */
    val classificationModel = clArgs.model match {
      case 0 =>
        new LogisticRegression()
          .setMaxIter(100)
          .setFitIntercept(true)

      case 1 =>
        new FactorizationMachine()
          .setLatentDimension(5)
          .setMaxIter(100)
          .setFitIntercept(true)
    }



    /*
     * Cross Validator parameter grid
     */
    val paramGrid = new ParamGridBuilder()
      .addGrid(classificationModel.regParam, Array(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3, 1e-2, 1e-1, 0.001341682))
      .addGrid(classificationModel.elasticNetParam, Array(0.95))
      .build()

    /*
     * Perform cross validation over the parameters
     */
    val cv = new CrossValidator()
      .setEstimator(classificationModel)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    /*
     * Run the grid search and pick up the best model
     */
    val bestModel = classificationModel match {
      case m: LogisticRegression =>
        cv.fit(trainingData)
          .bestModel.asInstanceOf[LogisticRegressionModel]

      case m: FactorizationMachine =>
        cv.fit(trainingData)
          .bestModel.asInstanceOf[FactorizationMachineModel]
    }

    /*
     * Predictions on the test data using the best model
     */
    val scoresPredictionsAndLabels = bestModel.transform(testData)
      .select("label", "probability", "prediction")
      .map{
        case Row(label: Double, probability: Vector, prediction: Double) =>
          (probability(1), prediction, label)
      }

    /*
     * Performance Metrics
     */
    // Area Under ROC
    val metrics = new BinaryClassificationMetrics(
      scoresPredictionsAndLabels
        .map(entry => (entry._1, entry._3))
    )

    println(s"AUC = ${metrics.areaUnderROC()}")

    bestModel match {
      case m: FactorizationMachineModel =>
        val mod = m.asInstanceOf[FactorizationMachineModel]
        println(s"Weights = ${mod.coefficients.intercept}; ${mod.coefficients.linear}")
      case _ => println("")
    }
    sc.stop()
  }
}
