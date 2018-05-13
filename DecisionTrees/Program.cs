using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using System.IO;
using Accord;
using Accord.Statistics;
using Accord.Statistics.Filters;
using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using System.Data;
using FastMember;
using Accord.Math;

namespace DecisionTrees
{
	class Program
	{
		static void Main(string[] args)
		{
			ReadStudentRecords();
			Console.ReadKey();
		}

		private static void ReadStudentRecords()
		{
			var csv = new CsvReader(File.OpenText("IntakesTrainingSet.csv"));
			var records = csv.GetRecords<StudentInfo>().ToList();

			//Convert data to table
			var data = new DataTable();
			using (var reader = ObjectReader.Create(records))
			{
				data.Load(reader);
			}

			// Loop through each column in data
			foreach (DataColumn column in data.Columns)
			{
				// Replace empty with underscore
				column.ColumnName = column.ColumnName.Replace(" ", "_");
			}

			// Create a new codification codebook to 
			// convert strings into integer symbols
			Codification codebook = new Codification(data);

			// Translate our training data into integer symbols using our codebook:
			DataTable symbols = codebook.Apply(data);
			int[][] inputs = symbols.ToJagged<int>(
				"was_aanwezig",
				//"gewogen_gemiddelde",
				"competenties",
				"capaciteiten",
				"intr_motivatie",
				"extr_motivatie",
				"is_mbo_deficient",
				"persoonlijk_bijspijker_advies",
				"Aanmelden_voor_verkort_opleidingstraject"
			);

			int[] outputs = symbols.ToMatrix<int>("advies").GetColumn(0);

			var id3 = new ID3Learning()
			{
				new DecisionVariable("was_aanwezig", 2),
				//new DecisionVariable("gewogen_gemiddelde", codebook.Columns.First(c => c.ColumnName == "gewogen_gemiddelde").NumberOfSymbols),
				new DecisionVariable("competenties", 10),
				new DecisionVariable("capaciteiten", 10),
				new DecisionVariable("intr_motivatie", 10),
				new DecisionVariable("extr_motivatie", 10),
				new DecisionVariable("is_mbo_deficient", 2),
				new DecisionVariable("persoonlijk_bijspijker_advies", 2),
				new DecisionVariable("Aanmelden_voor_verkort_opleidingstraject", 2)
			};

			DecisionTree tree = id3.Learn(inputs, outputs);

			//Now that we have a decision tree, load in the test set and test
			csv = new CsvReader(File.OpenText("IntakesTestSet.csv"));
			var testRecords = csv.GetRecords<StudentInfo>();

			foreach (StudentInfo record in testRecords)
			{
				

				//Transform the values of the test set into the same internal values used in the training set
				int[] query = codebook.Transform(new[,]
					{
						{ "was_aanwezig", record.was_aanwezig },
						//{ "gewogen_gemiddelde", record.gewogen_gemiddelde },
						{ "competenties", record.competenties },
						{ "capaciteiten", record.capaciteiten },
						{ "intr_motivatie", record.intr_motivatie },
						{ "extr_motivatie", record.extr_motivatie },
						{ "is_mbo_deficient", record.is_mbo_deficient },
						{ "persoonlijk_bijspijker_advies", record.persoonlijk_bijspijker_advies },
						{ "Aanmelden_voor_verkort_opleidingstraject", record.Aanmelden_voor_verkort_opleidingstraject },
					}
				);

				int predicted = tree.Decide(query);

				string answer;

				try
				{
					answer = codebook.Revert("advies", predicted);
				}
				catch(KeyNotFoundException)
				{
					Console.WriteLine($"Could not generate advice for student {record.studentnummer}");
					continue;
				}

				Console.WriteLine($"Student {record.studentnummer}: {answer}");
			}
		}
	}
}
