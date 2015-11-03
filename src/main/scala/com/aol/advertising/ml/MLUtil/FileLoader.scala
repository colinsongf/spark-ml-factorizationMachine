package com.aol.advertising.ml.MLUtil

/**
 * Project          : aol-ml
 * File             : FileLoader
 * Author           : Nikhil J Joshi (nikhil.joshi@teamaol.com)
 * Date Created     : 10/21/15
 *
 * Descriptor       : 
 * " The class usage goes here. "
 *
 */
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.feature.{StandardScalerModel, StandardScaler}
import org.apache.spark.sql.types.{StructField, StructType, StringType}
import org.apache.spark.sql.{Row, SQLContext, DataFrame}

import scala.collection.immutable.{HashSet, HashMap}
import scala.collection.mutable.{HashSet => MutableHashSet, Map => MutableMap, HashMap => MutableHashMap}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.log4j.Logger

object FileLoader {

  /*
   * Logger
   */
    val log = Logger.getLogger(getClass.getName)

  /*
   * Methods of standardizing datasets
   */
  object DataStandardizationMethod extends Enumeration {
    type DataStandardizationMethod = Value
    val none, withMean, withStd, withMeanAndStd = Value
  }

  /*
   * Placeholder for parameters used while loading a file
   */
  type LoaderParams = HashMap[String, Any]


  /*
   * Does the specified file exist
   */
  def isFound(filepath: String): Boolean =
    FileSystem.get(new Configuration()).globStatus(new Path(filepath)) match {
      case null => false
      case _ => true
    }

  /*
   * Extract campaign ID from the absolute path
   *
   * The campaign id is the NumSeq from the
   * first match of type campaignNumSeq
   */

  def extractCampaignID(filename: String): Option[Int] = {

    // Extract campaign ID
    val campainFileName = ".*(campaign)(\\d+).*".r.findAllIn(filename)

    // If found one
    if (campainFileName.hasNext) {
      campainFileName.next()
      Some(campainFileName.group(2).toInt)
    }
    else {
      log.warn(s"Unable to extract an integral value " +
        s"for CampaignID from the file name of $filename")
      None
    }
  }


  /*
   * CSV Files loader
   */
  implicit class CsvLoader(val sc: SparkContext) {

    class CsvData(val data: RDD[LabeledPoint],
                  val featureIndexMap: Map[String, Int],
                  val loaderParams: LoaderParams)

    import DataStandardizationMethod._

    /*
     * Load the CSV file as an RDD
     *
     * Input:
     * file:  csv file path
     * separator:  identifier/separator used to separate independent column values
     * withHeader: If the first row of the file to be parsed as a header (with column names)
     * applyLogTransform: If individual values to be log transformed
     */
    def loadCsvFileAsRDD(file: String,
                         separator: String = ",",
                         withHeader: Boolean = false,
                         applyLogTransform: Boolean = false,
                         standardizeData: DataStandardizationMethod = DataStandardizationMethod.none,
                         dropColumns: HashSet[String] = HashSet[String](),
                         selectColumns: HashSet[String] = HashSet[String](),
                         categoricalFeatures: HashSet[String] = HashSet[String](),
                         withFeatureIndexing: Map[String, Int] = Map[String, Int](),
                         withLoaderParams: LoaderParams = HashMap[String, Any]())
    : CsvData = {

      /*
       * Parameters used for loading this file
       */
      if (withLoaderParams.nonEmpty)
        log.info(s"With loading parameters specified using 'withLoaderParams' " +
          "any redundant separately specified parameter will be ignored.")
      val currentLoaderParams = MutableHashMap[String, Any](withLoaderParams.toSeq:_*)
      if ( !currentLoaderParams.contains("separator") )
        currentLoaderParams += "separator" -> separator
      if ( !currentLoaderParams.contains("withHeader") )
        currentLoaderParams += "withHeader" -> withHeader
      if ( !currentLoaderParams.contains("applyLogTransform") )
        currentLoaderParams += "applyLogTransform" -> applyLogTransform
      if ( !currentLoaderParams.contains("standardizeData") )
        currentLoaderParams += "standardizeData" -> standardizeData

      /*
       * Dropping columns is possible only with header information
       */
      if ( dropColumns.nonEmpty || selectColumns.nonEmpty )
        require(withHeader,
          "Dropping/selecting columns is not accurate without header information.")

      /*
       * Selecting columns has preference
       */
      if ( selectColumns.nonEmpty && dropColumns.nonEmpty )
        log.info(s"Both selection and dropping criterion for columns specified. " +
          "Column selection has a higher preference and " +
          "a column will be selected if its name appears in both criteria.")


      /*
       * The data file
       */
      val textFile = sc.textFile(file)
        .map(line => line.trim.split(currentLoaderParams("separator").asInstanceOf[String]))

      /*
       * Extract header into column names
       *
       * If file does not contain a header,
       * column index in the original file is used as a header
       */
      val header:Array[String] = currentLoaderParams("withHeader").asInstanceOf[Boolean] match {
        case false => (0 to textFile.first().length-1).map(_.toString).toArray
        case true => textFile.first()

      }
      // Drop the header from RDD
      val fileSansHeader = currentLoaderParams("withHeader").asInstanceOf[Boolean] match {
        case false => textFile
        case true =>
          textFile.mapPartitionsWithIndex(
            (i, iterator) =>
              if (i == 0 && iterator.hasNext) {
                iterator.next()
                iterator
              } else iterator
          )
      }

      /*
       * Column indices to be selected
       *
       * If selection criterion given use that as-is
       * else use the whole of header without the columns to be dropped
       */
      val columnsToBeSelected: Set[String] = selectColumns.nonEmpty match {
        case true => // select the columns
          selectColumns
        case false =>
          if (withFeatureIndexing.nonEmpty)
            withFeatureIndexing
              .map{ case(name, index) => name }
              .map(_.split("=").head)
              .toSet
          else {
            dropColumns.nonEmpty match {
              case true => // nothing to select, but drop columns
                header.filter(!dropColumns.contains(_)).toSet
              case false => // nothing to drop
                header.toSet
            }
          }
      }

      val selectColumnIndices =
        header
          .zipWithIndex
          .filter{ case(name, idx) => columnsToBeSelected.contains(name) }
          .map(_._2)
          .toSet


      /*
       * Final Categorical features and the respective
       * category to be muted, to avoid collinearity issues
       */
      // All the specified categorical features
      val allCategoricalFeatures: MutableHashSet[String] =
        MutableHashSet(categoricalFeatures.toSeq:_*)
      // Update the list with any extra feature mentioned in thea
      allCategoricalFeatures ++=
        withFeatureIndexing.keys
          .filter(_.contains("="))
          .map(_.split("="))
          .map(_.head)


      /*
       * Collect each record as (feat, value) pair if
       * (a) it is not to be dropped, and
       * (b) attach the value to feat=value if it is a categorical variable, and
       * (c) append to features set accumulator any new
       *
       */
      val featuresSetAcc =
        sc.accumulator(MutableHashSet[String]())(new MutableHashSetAccParam[String]())

      val sparseVW = fileSansHeader
        .map(record => {
          var label = 0.0
          try {
            label = record(0).toDouble
          } catch {
            case e: Exception => log.error(s"Error converting ${record(0)} to Double. " + e)
          }
          var index = 0
          val featuresAndvalues: Array[(String, Double)]  =
            record.tail
              .map(item => if(item.nonEmpty) item else "<EMPTYVAL>")
              .map(item  => {
                index += 1
                (index, item)
              })
              .filter{case (idx, value) => selectColumnIndices.contains(idx)}
              .map{
                case(idx, value) =>
                  allCategoricalFeatures.contains(header(idx)) match {
                    case true =>
                      val featureName = s"${header(idx)}=${value.toString}"
                      featuresSetAcc ++= MutableHashSet(featureName)
                      (featureName, 1.0)
                    case false =>
                      featuresSetAcc ++= MutableHashSet(header(idx))
                      val correctedValue =
                        if (value.contains("<EMPTYVAL>"))
                          scala.Double.MinPositiveValue
                        else value.toDouble + scala.Double.MinPositiveValue
                      (header(idx), if (currentLoaderParams("applyLogTransform").asInstanceOf[Boolean]) math.log(correctedValue) else correctedValue)
                  }
              }
          (label, featuresAndvalues)
        })


      /*
       * Features set
       */
      val (numberOfFeatures, featureIndexMap) = withFeatureIndexing.isEmpty match {
        case true => // No indexing was provided

          sparseVW.count()
          // To avoid collinearity issues
          // One category needs to be dropped out of each categorical feature
          val droppedCategory =
            featuresSetAcc.value
              .filter(_.contains("="))                    // only featureName=categoryID
              .map(_.split("="))                          // featureName, categoryID
              .map(array => (array.head, array.last))     // make tuple (featureName, categoryID)
              .toMap                                      // make map featureName -> categoryID .... confirms there is only one entry as featureName ->
              .map(entry => s"${entry._1}=${entry._2}")   // get back original featureName=categoryID
              .toSet                                      // for faster search

          (featuresSetAcc.value.size - droppedCategory.size,
            Map[String, Int](featuresSetAcc.value
            .filter( !droppedCategory.contains(_) )
            .zipWithIndex
            .toSeq: _*)
            )

        case false => // External feature index map is to be used
          (withFeatureIndexing.size, withFeatureIndexing)
      }

      val data = sparseVW.map { case(label: Double, featureValueArray: Array[(String, Double)]) =>
        val featureVector = Vectors.sparse(
          numberOfFeatures,
          featureValueArray
            .filter { case (featureName, value) => featureIndexMap.contains(featureName) }  // filter out those features not covered by the (externally provided) featureIndexMap
            .map { case (featureName, value) => (featureIndexMap.getOrElse(featureName, -1), value) }.toSeq
        )

        LabeledPoint(label, featureVector)
      }

      /*
       * Apply standardization if requested
       * Note: Do not standardize categorical features
       */
      val dataStdMethod = currentLoaderParams("standardizeData").asInstanceOf[DataStandardizationMethod]
      val withMean = dataStdMethod == DataStandardizationMethod.withMean ||
        dataStdMethod == DataStandardizationMethod.withMeanAndStd
      val withStd = dataStdMethod == DataStandardizationMethod.withStd ||
        dataStdMethod == DataStandardizationMethod.withMeanAndStd


      val standardizedData = dataStdMethod == none match {
        case true =>
          data
        case false =>
          val (catFeatMaskedMean, catFeatMaskedStd) =
          // is standardization mean and/or std provided?
            currentLoaderParams.contains("correctedStandardizationMean") &&
              currentLoaderParams.contains("correctedStandardizationStd") match {

              case false =>
                // Index to featureName map
                val indexedFeatureMap = featureIndexMap.map(_.swap)
                // A raw standardizer
                val dataStandardizer = new StandardScaler(withMean = withMean, withStd = withStd).fit(data.map(x => x.features))

                // Adjust/mask for categorical variabels
                (Vectors.sparse(numberOfFeatures,
                  dataStandardizer.mean.toSparse.indices
                    .filter(!indexedFeatureMap(_).contains("="))
                    .map(idx => (idx, dataStandardizer.mean(idx))).toSeq
                ).toDense,
                  Vectors.sparse(numberOfFeatures,
                    dataStandardizer.mean.toSparse.indices.map { idx =>
                      indexedFeatureMap(idx).contains("=") match {
                        case true => (idx, 1.0)
                        case false => (idx, dataStandardizer.std(idx))
                      }
                    }.toSeq
                  ).toDense
                  )

              case true =>
                (currentLoaderParams("correctedStandardizationMean").asInstanceOf[DenseVector],
                  currentLoaderParams("correctedStandardizationStd").asInstanceOf[DenseVector])
            }

          currentLoaderParams("correctedStandardizationMean") = catFeatMaskedMean
          currentLoaderParams("correctedStandardizationStd") = catFeatMaskedStd

          // Standardize data with this corrected standardizer
          data.map(point =>
            LabeledPoint(point.label,
              new StandardScalerModel(catFeatMaskedStd, catFeatMaskedMean)
                .setWithMean(withMean)
                .setWithStd(withStd)
                .transform(point.features.toDense).toSparse))
      }


      new CsvData(
        data = standardizedData,
        featureIndexMap = featureIndexMap,
        loaderParams = HashMap(currentLoaderParams.toSeq:_*)
      )
    }

  }

}
