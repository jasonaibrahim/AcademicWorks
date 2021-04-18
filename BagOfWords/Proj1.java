import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/*
 * This is the skeleton for CS61c project 1, Fall 2012.
 *
 * Contact Alan Christopher or Ravi Punj with questions and comments.
 *
 * Reminder:  DO NOT SHARE CODE OR ALLOW ANOTHER STUDENT TO READ YOURS.
 * EVEN FOR DEBUGGING. THIS MEANS YOU.
 *
 */
public class Proj1 {

    /** An Example Writable which contains two String Objects. */
    public static class StringPair implements Writable {
        /** The String objects I wrap. */
	private String a, b;

	/** Initializes me to contain empty strings. */
	public StringPair() {
	    a = b = "";
	}
	
	/** Initializes me to contain A, B. */
        public StringPair(String a, String b) {
            this.a = a;
	    this.b = b;
        }

        /** Serializes object - needed for Writable. */
        public void write(DataOutput out) throws IOException {
            new Text(a).write(out);
	    new Text(b).write(out);
        }

        /** Deserializes object - needed for Writable. */
        public void readFields(DataInput in) throws IOException {
	    Text tmp = new Text();
	    tmp.readFields(in);
	    a = tmp.toString();
	    
	    tmp.readFields(in);
	    b = tmp.toString();
        }

	/** Returns A. */
	public String getA() {
	    return a;
	}
	/** Returns B. */
	public String getB() {
	    return b;
	}
    }


  /**
   * Inputs a set of (docID, document contents) pairs.
   * Outputs a set of (Text, ObjectWritable) pairs.
   */
    public static class Map1 extends Mapper<WritableComparable, Text, Text, Text> {
        /** Regex pattern to find words (alphanumeric + _). */
        final static Pattern WORD_PATTERN = Pattern.compile("\\w+");

        private String targetGram = null;
        private int funcNum = 0;

        /*
         * Setup gets called exactly once for each mapper, before map() gets called the first time.
         * It's a good place to do configuration or setup that can be shared across many calls to map
         */
        @Override
        public void setup(Context context) {
            targetGram = context.getConfiguration().get("targetGram").toLowerCase();
	    try {
		funcNum = Integer.parseInt(context.getConfiguration().get("funcNum"));
	    } catch (NumberFormatException e) {
		/* Do nothing. */
	    }
        }
	
        /* Method to compute the number of words in a string */
        public int wordCount(String s)
        {
            int wc = 1;
            for(int i = 0; i < s.length(); i++)
            {
                if(s.charAt(i) == ' ')
                {
                    wc++;
                }
            }
            return wc;
        }

        /* Given an array list of distances, (index corresponding to a particular
         ngram) and treating each index across each list as a member of a set, this
         method will return a new array list that contains only the minimum value of the
         set, with all indices preserved. */
        public ArrayList<Double> minimumDistance(ArrayList<ArrayList> list)
        {
            ArrayList<Double> minList = new ArrayList<Double>();

            for(int i = 0; i < list.get(0).size(); i++)
            {
                TreeSet<Double> temp = new TreeSet<Double>();
                for(int j = 0; j < list.size(); j++)
                {
                    ArrayList l = list.get(j);
                    Object o = l.get(i);
                    Double d = (Double)o;
                    temp.add(d);
                }
                minList.add(temp.first());
            }
            return minList;
        }

        /* Given a list of n-grams and the index of the target word in the
         list, this method traverses the list and computes the distance
         between each n-gram and the target word. */
        public ArrayList<Double> makeDistanceList(ArrayList gramList, double index)
        {
            double zero = 0.0;
            ArrayList<Double> newList = new ArrayList<Double>();
            for(double k = 0; k < gramList.size(); k++)
            {
                if(k < index)
                {
                    newList.add(index-k);
                }
                else if(k == index)
                {
                    newList.add(zero);
                }
                else
                {
                    newList.add(k-index);
                }
            }
            return newList;
        }

        /* This method will create a list that contains the distances
         between ngrams (corresponding to index in list) and the target word.
         This method uses the helper functions above. */
        public ArrayList<Double> gramDistance(ArrayList l, String target)
        {
            boolean found = false;
            ArrayList<Double> gram_not_found_list = new ArrayList<Double>();
            ArrayList<Double> distanceList = new ArrayList<Double>();
            double i = 0;

            /* Find the index of the target words in the list of n-grams. If there
             are multiple target grams in the source, add the index of each one to
             the list. */
            for(i = 0; i<l.size(); i++)
            {
                if(l.get((int)i).equals(target))
                {
                    distanceList.add(i);
                    found = true;
                }
            }

            /* If we have not found the target gram in the list of grams
             we set all distances to positive infinity and return this list. */
            if(!found)
            {
                for(int t=0; t<l.size(); t++)
                {
                    gram_not_found_list.add(Double.POSITIVE_INFINITY);
                }
                return gram_not_found_list;
            }

            /* Now we create a list of lists, containing the distances
             for all ngrams and each occurance of the target word in the
             document */
            ArrayList<ArrayList> listOfDistances = new ArrayList();
            for(int h = 0; h < distanceList.size(); h++)
            {
                listOfDistances.add(makeDistanceList(l,distanceList.get(h)));
            }

            /* Now that we have a list of distances, we create a final list that
             contains just the minimum distances between each ngram and the target
             gram */
            ArrayList minList = new ArrayList();
            minList = minimumDistance(listOfDistances);

            return minList;
        }

        public ArrayList<String> makeListOfnGrams(Matcher matcher, int gramLength)
        {
            ArrayList<String> list = new ArrayList<String>();
            int lastIndex = 0;
            int i = 0;

            boolean test = true;

            parse:
            while(test)
            {
                boolean first_round = true;
                String ngram = new String();
                for(int k = 0; k < gramLength; k++)
                {
                    if(first_round)
                    {
                        if(!matcher.find(lastIndex))
                        {
                            test = false;
                            break parse;
                        }
                        lastIndex = matcher.end();
                        i = lastIndex;
                        ngram = ngram + matcher.group() + " ";
                        first_round = false;
                    }
                    else
                    {
                        if(!matcher.find(i))
                        {
                            test = false;
                            break parse;
                        }
                        ngram = ngram + matcher.group() + " ";
                        i = matcher.end();
                    }
                }
                ngram = ngram.substring(0,ngram.length()-1);
                ngram = ngram.toLowerCase();
                list.add(ngram);
            }
            return list;
        }

        @Override
        public void map(WritableComparable docID, Text docContents, Context context)
                throws IOException, InterruptedException {
                    
                    Matcher matcher = WORD_PATTERN.matcher(docContents.toString());
                    Func func = funcFromNum(funcNum);
                    
                    /* Find the length of the target gram */
                    int gramLength = wordCount(targetGram);
                    /* Make a list of all the ngrams that appear in the document */
                    ArrayList<String> list_of_ngrams = makeListOfnGrams(matcher, gramLength);
                    /* Make a list of the minimum distance of each ngram with respect to the target word */
                    ArrayList<Double> distanceList = gramDistance(list_of_ngrams,targetGram);
	    
                    for(int i=0; i < list_of_ngrams.size(); i++)
                    {
                        if(distanceList.get(i) == 0)
                        {
                            continue;
                        }
                        Double f_of_d = func.f(distanceList.get(i));
                        String outputPair = "" + Double.toString(f_of_d) + "," + "1";
                        Text value = new Text(outputPair);
                        Text key = new Text(list_of_ngrams.get(i));
                        context.write(key,value);
                    }

                }

        /** Returns the Func corresponding to FUNCNUM*/
        private Func funcFromNum(int funcNum) {
            Func func = null;
            switch (funcNum) {
                case 0:
                    func = new Func() {
                        public double f(double d) {
                            return d == Double.POSITIVE_INFINITY ? 0.0 : 1.0;
                        }
                    };
                    break;
                case 1:
                    func = new Func() {
                        public double f(double d) {
                            return d == Double.POSITIVE_INFINITY ? 0.0 : 1.0 + 1.0 / d;
                        }
                    };
                    break;
                case 2:
                    func = new Func() {
                        public double f(double d) {
                            return d == Double.POSITIVE_INFINITY ? 0.0 : Math.sqrt(d);
                        }
                    };
                    break;
                }
            return func;
        }
    }

    /** Here's where you'll be implementing your combiner. It must be non-trivial for you to receive credit. */
    public static class Combine1 extends Reducer<Text, Text, Text, Text> {

      @Override
      public void reduce(Text key, Iterable<Text> values,
              Context context) throws IOException, InterruptedException {
	  
          /* The idea here is to add up some of the f(d)'s and some of the instances of the 
           ngrams to help out my buddies in the reducers. */
          
          /* This variable is the partial sum of the f(d) */
          double partialS = new Double(0);
          
          /* This variable is the partial sum of the cardinality of G */
          int G = 0;
          
          String[] outputPair = new String[2];
          Iterator<Text> i = values.iterator();
	  
          while(i.hasNext())
          {
              Text val = i.next();
              String string_val = val.toString();
              outputPair = string_val.split(",");
              partialS += Double.parseDouble(outputPair[0]);
              G++;
	      }

          String outVal = "" + Double.toString(partialS) + "," + G;
          Text outputValue = new Text(outVal);
          context.write(key,outputValue);
      }
    }


    public static class Reduce1 extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values,
			   Context context) throws IOException, InterruptedException {
            
            /* The idea here is to reduce all the shits and then emit a pair
             like (co-occurance, ngram) */
            double S = new Double(0);
            int G = 0;
            String[] outputPair = new String[2];
            Iterator<Text> i = values.iterator();

            while(i.hasNext())
            {
                Text val = i.next();
                String string_val = val.toString();
                outputPair = string_val.split(",");
                S += Double.parseDouble(outputPair[0]);
                G += Integer.parseInt(outputPair[1]);
            }
            double magnitude_G = new Double(G);
            double C = new Double(0);
            if(S > 0)
            {
                C = ((S*(Math.pow(Math.log(S),3)))/magnitude_G);
            }
	    
            Text value = new Text(Double.toString(C));
            context.write(value,key);
		
        }
    }

    public static class Map2 extends Mapper<Text, Text, DoubleWritable, Text> {
  
        @Override
	    public void map(Text key, Text value, Context context)
	    throws IOException, InterruptedException {
	    
            /* This part was a little clever I suppose, I thought of it while I was eating
             a rice bowl at urbann turbann. I swear that place is the tits. Flip the sign on all
             the values, so that it gets sent out to the reducer in reverse sorted order. The trick
             here was to make sure it was a DoubleWritable. There must be something going on in the
             compareTo method for strings that doesnt make it sort right. So DoubleWritables and sall good*/
            String s = key.toString();
            Double d = Double.parseDouble(s);
            d *= -1;
            DoubleWritable dw = new DoubleWritable(d);
            context.write(dw,value);
        }
	    
    }

    public static class Reduce2 extends Reducer<DoubleWritable, Text, DoubleWritable, Text> {

      int n = 0;
      static int N_TO_OUTPUT = 100;

      /*
       * Setup gets called exactly once for each reducer, before reduce() gets called the first time.
       * It's a good place to do configuration or setup that can be shared across many calls to reduce
       */
      @Override
      protected void setup(Context c) {
        n = 0;
      }

        @Override
        public void reduce(DoubleWritable key, Iterable<Text> values,
                Context context) throws IOException, InterruptedException {
            
            /* Since we flipped the signs to get it in reverse sorted order, just flip the signs back
             and emit exactly what we want. This whole process is really magical. */
            Iterator<Text> i = values.iterator();
            while(i.hasNext())
            {
                Double d = key.get();
		if (d < 0)
		{
                    d *= -1;
		}
		if(d == -0)
		{
		    d = new Double(0);
		}
                DoubleWritable keyOut = new DoubleWritable(d);
                context.write(keyOut,i.next());
            }
        }
    }
    /*
     *  You shouldn't need to modify this function much. If you think you have a good reason to,
     *  you might want to discuss with staff.
     *
     *  The skeleton supports several options.
     *  if you set runJob2 to false, only the first job will run and output will be
     *  in TextFile format, instead of SequenceFile. This is intended as a debugging aid.
     *
     *  If you set combiner to false, neither combiner will run. This is also
     *  intended as a debugging aid. Turning on and off the combiner shouldn't alter
     *  your results. Since the framework doesn't make promises about when it'll
     *  invoke combiners, it's an error to assume anything about how many times
     *  values will be combined.
     */
    public static void main(String[] rawArgs) throws Exception {
        GenericOptionsParser parser = new GenericOptionsParser(rawArgs);
        Configuration conf = parser.getConfiguration();
        String[] args = parser.getRemainingArgs();

        boolean runJob2 = conf.getBoolean("runJob2", true);
        boolean combiner = conf.getBoolean("combiner", false);

        if(runJob2)
          System.out.println("running both jobs");
        else
          System.out.println("for debugging, only running job 1");

        if(combiner)
          System.out.println("using combiner");
        else
          System.out.println("NOT using combiner");

        Path inputPath = new Path(args[0]);
        Path middleOut = new Path(args[1]);
        Path finalOut = new Path(args[2]);
        FileSystem hdfs = middleOut.getFileSystem(conf);
        int reduceCount = conf.getInt("reduces", 32);

        if(hdfs.exists(middleOut)) {
          System.err.println("can't run: " + middleOut.toUri().toString() + " already exists");
          System.exit(1);
        }
        if(finalOut.getFileSystem(conf).exists(finalOut) ) {
          System.err.println("can't run: " + finalOut.toUri().toString() + " already exists");
          System.exit(1);
        }

        {
            Job firstJob = new Job(conf, "wordcount+co-occur");

            firstJob.setJarByClass(Map1.class);

	    /* You may need to change things here */
            firstJob.setMapOutputKeyClass(Text.class);
            firstJob.setMapOutputValueClass(Text.class);
            firstJob.setOutputKeyClass(Text.class);
            firstJob.setOutputValueClass(Text.class);
	    /* End region where we expect you to perhaps need to change things. */

            firstJob.setMapperClass(Map1.class);
            firstJob.setReducerClass(Reduce1.class);
            firstJob.setNumReduceTasks(reduceCount);


            if(combiner)
              firstJob.setCombinerClass(Combine1.class);

            firstJob.setInputFormatClass(SequenceFileInputFormat.class);
            if(runJob2)
              firstJob.setOutputFormatClass(SequenceFileOutputFormat.class);

            FileInputFormat.addInputPath(firstJob, inputPath);
            FileOutputFormat.setOutputPath(firstJob, middleOut);

            firstJob.waitForCompletion(true);
        }

        if(runJob2) {
            Job secondJob = new Job(conf, "sort");

            secondJob.setJarByClass(Map1.class);
	    /* You may need to change things here */
            secondJob.setMapOutputKeyClass(DoubleWritable.class);
            secondJob.setMapOutputValueClass(Text.class);
            secondJob.setOutputKeyClass(DoubleWritable.class);
            secondJob.setOutputValueClass(Text.class);
	    /* End region where we expect you to perhaps need to change things. */

            secondJob.setMapperClass(Map2.class);
            if(combiner)
              secondJob.setCombinerClass(Reduce2.class);
            secondJob.setReducerClass(Reduce2.class);

            secondJob.setInputFormatClass(SequenceFileInputFormat.class);
            secondJob.setOutputFormatClass(TextOutputFormat.class);
            secondJob.setNumReduceTasks(1);


            FileInputFormat.addInputPath(secondJob, middleOut);
            FileOutputFormat.setOutputPath(secondJob, finalOut);

            secondJob.waitForCompletion(true);
        }
    }

}
