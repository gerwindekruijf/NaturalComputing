/* 	================= DESCRIPTION ===================== */
/* This text is printed on the HTML page. */
/** @file
 * Cells sorting themselves through differential adhesion.
 **/

/* 	================= DECLARE CUSTOM METHODS ===================== */
/* 	If no custom methods are defined, the drawing/initialisation/output 
	functions of the CPM.Simulation class are used. */

/* START METHODS OBJECT Do not remove this line */
/* 	The following functions are defined below and will be added to
	the simulation object.*/
    let custommethods = {
        initializeGrid : initializeGrid
    }
    /* END METHODS OBJECT Do not remove this line */
    
    
    /* ================= WRITE CUSTOM METHODS ===================== */
    
    /* START METHODS DEFINITION Do not remove this line */
    
    /* The following custom methods will be added to the simulation object
    below. */
    function initializeGrid(){
	
        // add the GridManipulator if not already there and if you need it
        if( !this.helpClasses["gm"] ){ this.addGridManipulator() }
    
        // Seed hier op een vaste locatie (in een grid) cellen van type 2
        for(i = 35; i < 200; i = i + 140){
            for(j = 35; j< 200; j = j + 140){
                this.gm.seedCellAt(2,[i,j])
            }
        }
        // Densely packed cellen generen 
        for(k = 0; k < 10; k++){
            this.gm.seedCell(1)
        }
    }
    
    /* END METHODS DEFINITION Do not remove this line */
    
    
    
    /* ================= CONFIGURATION ===================== */
    
    /* Do not remove this line: START CONFIGURATION */
    /*	----------------------------------
        CONFIGURATION SETTINGS
        ----------------------------------
    */
    let config = {
    
        // Grid settings
        ndim : 2,
        field_size : [200,200],
        
        // CPM parameters and configuration
        conf : {
            // Basic CPM parameters
            torus : [true,true],						// Should the grid have linked borders?
            seed : 1,							// Seed for random number generation.
            T : 20,								// CPM temperature
    
            // Constraint parameters. 
            // Mostly these have the format of an array in which each element specifies the
            // parameter value for one of the cellkinds on the grid.
            // First value is always cellkind 0 (the background) and is often not used.
            
            // Adhesion parameters:
            J : [ [0, 20, 0], [20, 0, 0], [0, 0, 0] ],
            
            // VolumeConstraint parameters
            // VolumeConstraint importance per cellkind
            // Target volume of each cellkind
            LAMBDA_V : [0,50,3],
            V : [0,200,100],	
            
            //PerimeterConstraint parameters
            //PerimeterConstraint importance per cellkind
            //Targer Perimeter
            LAMBDA_P  : [0,2,2],
            P : [0,180,35],

                    // ActivityConstraint parameters
            LAMBDA_ACT : [0,200,0],				// ActivityConstraint importance per cellkind
            MAX_ACT : [0,80,0],					// Activity memory duration per cellkind
            ACT_MEAN : "geometric"				// Is neighborhood activity computed as a
                                                // "geometric" or "arithmetic" mean?	
        
                
        },
        
        // Simulation setup and configuration
        simsettings : {
        
            // Cells on the grid
            NRCELLS : [1,1],					// Number of cells to seed for all
            // non-background cellkinds.
        
            // Runtime etc
            BURNIN : 500,
            RUNTIME : 1000,
            RUNTIME_BROWSER : "Inf",
            
            // Visualization
            CANVASCOLOR : "eaecef",
            CELLCOLOR : ["FF0000","000000"],
            ACTCOLOR : [true,false],			// Should pixel activity values be displayed?
            SHOWBORDERS : [true,true],				// Should cellborders be displayed?
            zoom : 2,							// zoom in on canvas with this factor.
            
            // Output images
            SAVEIMG : true,					// Should a png image of the grid be saved
            // during the simulation?
            IMGFRAMERATE : 1,					// If so, do this every <IMGFRAMERATE> MCS.
            SAVEPATH : "output/img/obstacle_sim",				// ... And save the image in this folder.
            EXPNAME : "Obstacle_sim",					// Used for the filename of output images.
            
            // Output stats etc
            STATSOUT : { browser: false, node: true }, // Should stats be computed?
            LOGRATE : 10							// Output stats every <LOGRATE> MCS.
    
        }
    }
    /*	---------------------------------- */
    /* Do not remove this line: END CONFIGURATION */
    