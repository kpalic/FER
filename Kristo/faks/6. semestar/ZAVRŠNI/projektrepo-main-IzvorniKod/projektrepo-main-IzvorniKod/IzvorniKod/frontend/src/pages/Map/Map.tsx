import React, {useState, useEffect, useRef, useCallback} from 'react'
import ReactMapGL, {Marker, Popup} from "react-map-gl";

import './Map.css'
import { searchEngine } from './SearchEngine';

//dodavanje lokacije
import * as Yup from 'yup'
import { useFormik } from 'formik'
import { useAuth } from '../../auth/authContext'

//Modeli
import { Business } from '../../models/Business.model';
import { Location, LocationType } from '../../models/Location.model';

//modeli za get
import { MapData } from '../../models/MapData.model';

import { FormControl, MenuItem, TextField, RadioGroup, Radio, Button, Hidden } from '@mui/material';
import Select from '@mui/material/Select';
import { UserRole } from '../../models/User.model';
import { AxiosInstance } from '../../auth/AxiosInstance';

const geocodingURL = "https://api.mapbox.com/geocoding/v5/mapbox.places/";
const TOKEN = "pk.eyJ1IjoicDNyazRuIiwiYSI6ImNsYWF2cHVzaDA2czYzd29hNXB2bWtscmIifQ.pyojsfUoaSU6XIZtXpwwew";

//Daje list best matching rezultata za navedenu adresu, ukljucuje koordinate
const addrToCoord = async (address : String | null) => 
{
  if(!address) return [];

  address = address.replaceAll(" ","-");
  const res = await fetch(geocodingURL + address + ".json?access_token=" + TOKEN);
  const data = await res.json();
  
  if(data.features.length > 0) return data.features[0].center;
  return [];
};

const addrsToCoords = async (addresses : String[]) => 
{
    addresses = addresses.map(address => address.replaceAll(" ","-"));
    var coords : any = [];

    for (const address of addresses) {
        const res = await fetch(geocodingURL + address + ".json?access_token=" + TOKEN);
        const data = await res.json();
        if(!data) throw new Error("Wrong address");
        coords.push(data.features[0].center);
    }
    
    return coords;
};

//-----------------------------------------------------------------------

 const lokacijaType = {} as Location;
 const obrtType = {...{} as Business, longitude : 0, latitude : 0};
 const ratedType = {} as Location;

const Map= ()=>{
    //get data from backend--------------------------------------------
    const [mapData, setMapData] = useState<MapData>();
    const [updateMap,setUpdateMap]=useState(false)

    useEffect(() => {
      AxiosInstance.get("/map")
        .then((res) => {
          console.log(res.data);
          setMapData(res.data);
        })
        .catch((err) => console.log(err.response.data.message));
    }, [updateMap]);
    

    //Map Reference----------------------------------------------------------
    const [mapRef,setMapRef] = useState<any>();

    //Current position Marker-------------------------------------------
    const [currentLocationMarker, setCurrentLocationMarker] = useState<any>(undefined);

    //Map callback -> called to center the map when entering page for the first time or if location has already been granted---------------------
    const onRefChange = useCallback((node : any) => 
    {
        if(node) 
        {
          navigator.geolocation.getCurrentPosition
          (
              (pos) => 
              {
                  console.log("Setting center to: ", pos.coords);
                  node.flyTo({
                    center: [pos.coords.longitude, pos.coords.latitude],
                    zoom: 15,
                    duration: 1500,
                    essential: true
                  });
                  setMapRef(node);

                  if(!currentLocationMarker)
                    setCurrentLocationMarker(() =>
                    (<Marker
                    longitude={pos.coords.longitude}
                    latitude={pos.coords.latitude}
                    anchor="center"
                    onClick={(event) => 
                    {
                      event.originalEvent.stopPropagation();
                      setPopupInfoBusiness(undefined);
                      setPopupInfoLocation(undefined);
                      node.flyTo({
                        center: [pos.coords.longitude, pos.coords.latitude],
                        duration: 1500,
                        essential: true,
                        zoom: 15
                      });
                      
                    }}
                    >
                      <img src={require('./../../assets/markeri/Home.png')} />
                    </Marker>))
              },
                (err) => 
              {
                  console.log("Get location error: ", err);
                  setMapRef(node);
              },
              { enableHighAccuracy: true, timeout: 5000, maximumAge: 1000 }
          );
          // node.doubleClickZoom.disable();
        }
        else setMapRef(null);
    }, []);

    //Lists of markers--------------------------------------------------------
    const [locationMarkers, setLocationMarkers] = useState<typeof lokacijaType[]>([]);
    const [businessMarkers, setBusinessMarkers] = useState<typeof obrtType[]>([]);
    const [ratedMarkers, setRatedMarkers] = useState<typeof ratedType[]>([]);
    
    //Setting lists of markers
    useEffect(() => 
    {
        if(mapData == undefined && !mapData) return;

        setLocationMarkers(mapData.locations);

       setRatedMarkers(mapData.positiveRatedLocations)
       ratedMarkers.concat(mapData.negativeRatedLocations)

        var obrtii = mapData.business;

        var addresses : String[] = obrtii.map(obrt => obrt.businessAddress + " " + obrt.businessCity);
        if(addresses.length == 0) return;

        addrsToCoords(addresses)
        .then(coords => 
        {
            var businessMarkersTmp : typeof obrtType[] = [];
            
            for(var i = 0; i < obrtii.length; i++)
                businessMarkersTmp.push({...obrtii[i], longitude: coords[i][0], latitude: coords[i][1], promotionDuration:""});
            
            setBusinessMarkers(businessMarkersTmp);
        })
        .catch(err => console.log(err));

    }, [mapData]);

     const [shownBusiness, setShownBusiness]=useState<typeof businessMarkers>()
     const [shownBusinessBackUp, setShownBusinessBackUp]=useState<typeof businessMarkers>()
     useEffect(()=>{
      console.log(shownBusiness)
        setShownBusiness([]);
         setShownBusiness(businessMarkers.filter(business=>{
          return business.promotionStart!=null
         }))
         setShownBusinessBackUp(businessMarkers.filter(business=>{
           return (business.id===1 || business.id===3 || business.id===5 || business.id===7 || business.id===9)
         }))
     },[businessMarkers])

     useEffect(()=>{
      console.log(shownBusiness?.length)
      shownBusiness?.length===0? setShownBusiness(shownBusinessBackUp):(<></>)
     },[shownBusiness])

    //Subset from markers which will be shown when searching, DEFAULT=>ALL-------------------
    const [searchLocationSubset, setSearchLocationSubset] = useState<Set<String>>(new Set(["BEACH", "PARK", "RESTAURANT", "COFFEE_BAR", "OTHER"]));
    const [searchBusinessSubset, setSearchBusinessSubset] = useState<Set<String>>(new Set(["SHOP", "SALON", "DAYCARE", "VET", "OTHER"]));

    //Filters------------------------------------------------------------------------
    const [locationFilters, setLocationFilters] = useState<any>();
    const [businessFilters, setBusinessFilters] = useState<any>();

    const engTohrv : any = 
    {
      "BEACH" : "PLAŽA",
      "PARK"  : "PARK",
      "RESTAURANT" : "RESTORAN",
      "COFFEE_BAR" : "KAFIĆ",
      "OTHER" : "OSTALO",
      "SHOP" : "TRGOVINA",
      "SALON" : "SALON",
      "DAYCARE" : "VRTIĆ",
      "VET" : "VETERINAR"
    }

    useEffect(() =>
    {
        setLocationFilters(["BEACH", "PARK", "RESTAURANT", "COFFEE_BAR", "OTHER"].map((locationFilter) =>
        <div className="Filter" key = {locationFilter}>
            <input type={"checkbox"} checked={searchLocationSubset.has(locationFilter)} onChange={(e) => 
                {
                    let tmpLocationTypes = new Set(searchLocationSubset);

                    if(e.currentTarget.checked) tmpLocationTypes.add(locationFilter);
                    else tmpLocationTypes.delete(locationFilter);

                    setSearchLocationSubset(tmpLocationTypes);
                }}></input>
            <p>{engTohrv[locationFilter]}</p>
        </div>));
    }, [searchLocationSubset]);


    useEffect(() =>
    {
        setBusinessFilters(["SHOP", "SALON", "DAYCARE", "VET", "OTHER"].map((businessFilter) => 
        <div className="Filter" key = {businessFilter}>
            <input type={"checkbox"} checked={searchBusinessSubset.has(businessFilter)} onChange={(e) => 
                {
                    let tmpBusinessTypes = new Set(searchBusinessSubset);
                    if(e.currentTarget.checked) tmpBusinessTypes.add(businessFilter);
                    else tmpBusinessTypes.delete(businessFilter);

                    setSearchBusinessSubset(tmpBusinessTypes);
                }}></input>
            <p>{engTohrv[businessFilter]}</p>
        </div>))
    }, [searchBusinessSubset]);

    //Search Results--------------------------------------------------------------------
    const searchBarRef = useRef<any>();
    const [searchResults, setSearchResults] = useState<any>([]);

    //Uses current list of locations and businesses while watching for filters, updates search results-------------------
    const getSearchResults = (e : any) =>
    {
        if(e == undefined) return;
        var val = (typeof e != "string") ? e.target.value : e;

        let locs : typeof lokacijaType[] = searchEngine(locationMarkers, '@name:' + val);
        let buss : typeof obrtType[] = searchEngine(businessMarkers, "@businessName:" + val);
        let resArr : any[] = [...locs, ...buss];

        resArr = resArr.filter
        (
            (searchVal) => 
            (searchVal.type && searchLocationSubset.has(searchVal.type)) ||
            (searchVal.businessType && searchBusinessSubset.has(searchVal.businessType))
        )

        setSearchResults(() => resArr.map((searchValue, index) => (
            <div className="SearchResult" onClick=
            {
                e2 =>
                {
                    if(mapRef) mapRef.flyTo
                    ({
                        center: [searchValue.longitude, searchValue.latitude],
                        zoom: 15,
                        essential: true,
                        duration: 1500
                    });

                    if(searchValue.name) 
                    {
                        setPopupInfoLocation(searchValue);
                        setPopupInfoBusiness(undefined);
                    }
                    else
                    {
                        setPopupInfoBusiness(searchValue);
                        setPopupInfoLocation(undefined);
                    }
                }
            }>
                {String(searchValue.name ? searchValue.name : searchValue.businessName)}
            </div>
        )));
    };

    //Updates search results when checking markers-----------------------
    useEffect(() =>
    {
        if(searchBarRef) getSearchResults(searchBarRef.current.value);
    }, [searchLocationSubset, searchBusinessSubset]);

    //add new location
    const [coordLng, setCoordLng] = useState(0);
    const [coordLat, setCoordLat] = useState(0);
    const auth = useAuth();
    const [showNewLocation, setShowNewLocation] = useState(false);
    const [alreadyRated, setAlreadyRated]=useState<string[]>([]);

    const newLocationSchema = Yup.object().shape({
      longitude: Yup.number(),
      latitude: Yup.number(),
      name: Yup.string().required("Obavezno"),
      type: Yup.string().required("Obavezno"),
      rating: Yup.string().required("Obavezno"),
    });

    const formik = useFormik({
      initialValues: {
        longitude: 0.0,
        latitude: 0.0,
        name: "",
        type: "",
        rating: "",
      },
      validationSchema: newLocationSchema,
      onSubmit: (values, { setErrors }) => {
        auth
          .addLocation(
            values.longitude,
            values.latitude,
            values.name,
            values.type,
            values.rating
          )
          .then((res) => {
            console.log(res);
            const pricuva = {...values, rating:values.rating==="POSITIVE"?1:0, votesSum:1,positiveVotes:values.rating==="POSITIVE"?1:0} as Location;
            setLocationMarkers(locationMarkers=>[...locationMarkers,pricuva])
            setAlreadyRated(alreadyRated=>[...alreadyRated,values.name])
            setRatedMarkers(ratedMarkers=>[...ratedMarkers,pricuva])
            setShowNewLocation(false)
            setErrorAddLocation("")
            formik.resetForm();
          })
          .catch((err) => {
            console.log(err.response.data.message);
            setErrorAddLocation(err.response.data.message)
          });
      },
      
    });

    //change rating of existing location
    const changeRatingSchema = Yup.object().shape({
      id: Yup.number(),
      rating: Yup.string(),
      name:Yup.string()
    });
    const formikRating = useFormik({
      initialValues: { id:0, rating: "", name:"" },
      validationSchema: changeRatingSchema,
      onSubmit: (values, { setErrors }) => {
        auth
          .changeLocationRating(values.id, values.rating)
          .then((res) => {console.log(res)
            setErrorAddLocation("")
            setAlreadyRated(alreadyRated=>[...alreadyRated,values.name])})
          .catch((err) => {
            console.log(err.response.data.message);
            console.log(err);
            setErrorAddLocation(err.response.data.message)
          });
          //updateaj mapData
          setUpdateMap(true)
      },
    });

    //Pop-ups------------------------------------------------------------------------
    const [popupInfoLocation, setPopupInfoLocation] = useState<typeof lokacijaType>();
    const [popupInfoBusiness, setPopupInfoBusiness] = useState<typeof obrtType>();

    //Removes location popup if selected popup gets unchecked 
    useEffect(() =>
    {
        if(popupInfoLocation && !searchLocationSubset.has(popupInfoLocation.type)) setPopupInfoLocation(undefined);
    }, [searchLocationSubset]);

    //Removes business popup if selected popup gets unchecked 
    useEffect(() =>
    {
        if(popupInfoBusiness && !searchBusinessSubset.has(popupInfoBusiness.businessType)) setPopupInfoBusiness(undefined);
    }, [searchBusinessSubset]);

    //Location and business marker elements--------------------------------------
    const [markerElementsLocation, setMarkerElementsLocation] = useState<any>();
    const [markerElementsBusiness, setMarkerElementsBusiness] = useState<any>();

    //Updating location Marker elements
    useEffect(() =>
    {
        var subsetLocations = locationMarkers.filter((location) => searchLocationSubset.has(location.type));

        setMarkerElementsLocation(subsetLocations.map((location : typeof lokacijaType, index) => 
        (
            <Marker
              key={`locationMarker-${index}`}
              longitude={location.longitude}
              latitude={location.latitude}
              anchor="bottom"
               onClick={e => {
                setPopupInfoLocation(location);
                setPopupInfoBusiness(undefined);
                if(mapRef)
                  mapRef.flyTo({
                    center: [location.longitude, location.latitude],
                    zoom: 15,
                    essential: true,
                    duration: 1500
                  });
                e.originalEvent.stopPropagation();
               }}
               >
                {(()=>{
                  if(String(location.type)==="BEACH"){
                    if(location.rating>=0.8){
                      return(<img src={require('./../../assets/markeri/Anchor_zeleni.png')}/>)
                    }else if(location.rating>=0.4){
                      return(<img src={require('./../../assets/markeri/Anchor_zuti.png')}/>)
                    }else{ return(<img src={require('./../../assets/markeri/Anchor_crveni.png')}/>)}
                  }else if(String(location.type)==="PARK"){
                    if(location.rating>=0.8){
                      return(<img src={require('./../../assets/markeri/BaseballBall_zeleni.png')}/>)
                    }else if(location.rating>=0.4){
                      return(<img src={require('./../../assets/markeri/BaseballBall_zuti.png')}/>)
                    }else{ return(<img src={require('./../../assets/markeri/BaseballBall_crveni.png')}/>)}
                  }else if(String(location.type)==="RESTAURANT"){
                    if(location.rating>=0.8){
                      return(<img src={require('./../../assets/markeri/Fork&Knife_zeleni.png')}/>)
                    }else if(location.rating>=0.4){
                      return(<img src={require('./../../assets/markeri/Fork&Knife_zuti.png')}/>)
                    }else{ return(<img src={require('./../../assets/markeri/Fork&Knife_crveni.png')}/>)}
                  }else if(String(location.type)==="COFFEE_BAR"){
                    if(location.rating>=0.8){
                      return(<img src={require('./../../assets/markeri/Coffee_zeleni.png')}/>)
                    }else if(location.rating>=0.4){
                      return(<img src={require('./../../assets/markeri/Coffee_zuti.png')}/>)
                    }else{ return(<img src={require('./../../assets/markeri/Coffee_crveni.png')}/>)}
                  }else if(String(location.type)==="OTHER"){
                    if(location.rating>=0.8){
                      return(<img src={require('./../../assets/markeri/List_zelena.png')}/>)
                    }else if(location.rating>=0.4){
                      return(<img src={require('./../../assets/markeri/List_zuti.png')}/>)
                    }else{ return(<img src={require('./../../assets/markeri/List_crveni.png')}/>)}
                  }
                })()}
                
            </Marker>

        )));
    }, [locationMarkers, searchLocationSubset, mapRef]);

    //Updating business Marker elements
    useEffect(() =>
    {
        var subsetBusinesses = businessMarkers.filter((business) => searchBusinessSubset.has(business.businessType));

        setMarkerElementsBusiness(subsetBusinesses.map((business : typeof obrtType, index) =>
        (
            <Marker
                key={`businessMarker-${index}`}
                longitude={business.longitude} 
                latitude={business.latitude}
                anchor="bottom"
                onClick={e => {
                setPopupInfoBusiness(business);
                setPopupInfoLocation(undefined);
                if(mapRef)
                  mapRef.flyTo({
                    center: [business.longitude, business.latitude],
                    zoom: 15,
                    essential: true,
                    duration: 1501
                  });
                e.originalEvent.stopPropagation();
                }}
            >
               {(()=>{
                 if(business.businessType==="SHOP"){
                  if(business.promotionStart!=null){return(<div className="popupCrown"><img className='crown-image' src={require('./../../assets/markeri/crown.png')}/><img src={require('./../../assets/markeri/ShoppingBasket_blue.png')}/></div>)}
                  return(<img src={require('./../../assets/markeri/ShoppingBasket_blue.png')}/>)
                 }else if(business.businessType==="SALON"){
                  if(business.promotionStart!=null){return(<div className="popupCrown"><img className='crown-image' src={require('./../../assets/markeri/crown.png')}/><img src={require('./../../assets/markeri/Cut_blue.png')}/></div>)}
                  return(<img src={require('./../../assets/markeri/Cut_blue.png')}/>)
                 }else if(business.businessType==="VET"){
                  if(business.promotionStart!=null){return(<div className="popupCrown"><img className='crown-image' src={require('./../../assets/markeri/crown.png')}/><img src={require('./../../assets/markeri/Pills_blue.png')}/></div>)}
                  return(<img src={require('./../../assets/markeri/Pills_blue.png')}/>)
                 }else if(business.businessType==="DAYCARE"){
                  if(business.promotionStart!=null){return(<div className="popupCrown"><img className='crown-image' src={require('./../../assets/markeri/crown.png')}/><img src={require('./../../assets/markeri/School_blue.png')}/></div>)}
                  return(<img src={require('./../../assets/markeri/School_blue.png')}/>)
                 }else{
                  if(business.promotionStart!=null){return(<div className="popupCrown"><img className='crown-image' src={require('./../../assets/markeri/crown.png')}/><img src={require('./../../assets/markeri/Suitcase_blue.png')}/></div>)}
                  return(<img src={require('./../../assets/markeri/Suitcase_blue.png')}/>)
                 }
                })()}

            </Marker>
        )));
    }, [businessMarkers, searchBusinessSubset,mapRef]);

    const [centeredLocationMarker, setCenteredLocationMarker] = useState<any>(undefined);

    const[errorAddLocation, setErrorAddLocation]=useState("")

    //Returned component----------------------------------------------------
    return (
      <div className="Container">
        <link
          href="https://api.mapbox.com/mapbox-gl-js/v2.6.1/mapbox-gl.css"
          rel="stylesheet"
        />

        {/* 1. BIG DIV -> MAP */}
        <div className="MapContainer">
          <ReactMapGL
            id="ReactMapGL"
            ref={onRefChange}
            mapboxAccessToken={TOKEN}
            mapStyle="mapbox://styles/mapbox/streets-v9"
            style={{
              margin: "3%",
              width: "100%",
              minHeight: "600px",
              height: "100%",
              borderRadius: "30px",
              border: "5px solid white",
            }}
            attributionControl={false}
            onResize={(event) => {}}
            initialViewState={{
              longitude: 15.966568,
              latitude: 45.815399,
              zoom: 12,
            }}
            //za dodavanje nove lokacije
            onClick={(e) => {
              setShowNewLocation(true);
              setCoordLat(e.lngLat.lat)
              setCoordLng(e.lngLat.lng)
              setErrorAddLocation("")
              formik.values.latitude = e.lngLat.lat;
              formik.values.longitude = e.lngLat.lng;
              if(mapRef && JSON.parse(sessionStorage.getItem("role") as string) ===
              UserRole.ROLE_USER)
              {
                mapRef.flyTo({
                  center: [e.lngLat.lng, e.lngLat.lat],
                  zoom: 15,
                  essential: true,
                  duration: 1500
                });
              }
            }}

          >
            {centeredLocationMarker}
            {markerElementsLocation}
            {markerElementsBusiness}
            {currentLocationMarker}

            {/* POPUP LOKACIJE */}
            {popupInfoLocation && (
              <Popup
                anchor="bottom"
                longitude={Number(popupInfoLocation.longitude)}
                latitude={Number(popupInfoLocation.latitude)}
                closeOnClick={true}
                onOpen={(e)=>{setShowNewLocation(false)
                            setErrorAddLocation("")}}
                onClose={() => {setPopupInfoLocation(undefined); setErrorAddLocation("")}}
              >
                <div className="popup-container">
                  {/* pozadina popupa ovisi o ratingu lokacije */}
                  <div
                    className={(() => {
                      if (popupInfoLocation.rating >= 0.8) {
                        return "popup-container-green";
                      } else if (popupInfoLocation.rating >= 0.4) {
                        return "popup-container-yellow";
                      } else {
                        return "popup-container-red";
                      }
                    })()}
                  >
                    <div className="popup-name">
                      {popupInfoLocation.name}
                    </div>
                    <div className="popup-item">
                    {(() => {
                      if (popupInfoLocation.type === LocationType.BEACH) {return "Plaža";
                      } else if (popupInfoLocation.type === LocationType.COFFEE_BAR) {return "Kafić";
                      } else if (popupInfoLocation.type === LocationType.PARK) {return "Park";
                      } else if (popupInfoLocation.type === LocationType.RESTAURANT) {return "Restoran";
                      } else {return "Ostalo";}
                    })()}
                    </div>
                    <div className="popup-item">
                      {(popupInfoLocation.rating * 100).toFixed(0)}% korisnika smatra ovu
                      lokaciju prikladnom
                    </div>

                    {/* ako postoji korisnik koji ima ulogu USER i nije vec rateo lokaciju */}
                    {JSON.parse(sessionStorage.getItem("role") as string) ===
                      UserRole.ROLE_USER &&
                    !mapData?.positiveRatedLocations.filter(
                      (rated) => rated.name === popupInfoLocation.name
                    ).length &&
                    !mapData?.negativeRatedLocations.filter(
                      (rated) => rated.name === popupInfoLocation.name
                    ).length &&
                    !alreadyRated.filter(
                      (rated) => rated === popupInfoLocation.name
                    ).length ? (
                      <>
                        <form
                          className="changeRatingContainer"
                          onSubmit={formikRating.handleSubmit}
                          // setAlreadyRated(alreadyRated=>[...alreadyRated,popupInfoLocation.name])
                        >
                          <FormControl className="addLocFormControl">
                          <div style={{visibility:"hidden", height:"0px"}}> {formikRating.values.id=popupInfoLocation.id}{formikRating.values.name=popupInfoLocation.name}</div>
                          {errorAddLocation.length?(<div style={{color:"red"}}>{errorAddLocation}</div>):( <div style={{ paddingTop: "1px" }}>
                              Dodajte svoje mišljenje o lokaciji:
                            </div>)}
                            <RadioGroup
                              name="rating"
                              className="radioGroupAddLocation"
                              value={formikRating.values.rating}
                              onChange={formikRating.handleChange}
                            >
                              <Radio
                                value="POSITIVE"
                                className="radioAddLocation"
                                size="small"
                                style={{ paddingLeft: "0px" }}
                              />
                              <div>Pozitivno</div>
                              <Radio
                                value="NEGATIVE"
                                className="radioAddLocation"
                                size="small"
                              />
                              <div>Negativno</div>
                            </RadioGroup>
                          </FormControl>
                          <Button
                            style={{ backgroundColor: "var(--darkBlue)" }}
                            variant="contained"
                            type="submit"
                            // onClick={(e)=>{setAlreadyRated(alreadyRated=>[...alreadyRated,popupInfoLocation.name]); 
                            //               formikRating.handleSubmit}}
                          >
                            Dodaj mišljenje!
                          </Button>
                        </form>
                      </>
                    ) : (
                      <></>
                    )}
                  </div>
                </div>
              </Popup>
            )}

            {/* POPUP OBRTA */}
            {popupInfoBusiness && (
              <Popup
                anchor="bottom"
                longitude={Number(popupInfoBusiness.longitude)}
                latitude={Number(popupInfoBusiness.latitude)}
                closeOnClick={true}
                onClose={() => setPopupInfoBusiness(undefined)}
                onOpen={(e)=>{setShowNewLocation(false); setErrorAddLocation("")}}
                
              >
                <div className="popup-container">
                  <div className='popup-container-blue'>
                  <div className="popup-name">
                    {popupInfoBusiness.businessName}
                  </div>
                  <div className="popup-item">
                  {(() => {
                      if (popupInfoBusiness.businessType === "VET") {return "Veterinarska ordinacija";
                      } else if (popupInfoBusiness.businessType === "SHOP") {return "Trgovina za pse";
                      } else if (popupInfoBusiness.businessType === "SALON") {return "Salon za uređivanje psa";
                      } else if (popupInfoBusiness.businessType === "DAYCARE") {return "Vrtić za pse";
                      } else {return "Ostalo";}
                    })()}
                  </div>
                  <div className="popup-item">
                    {popupInfoBusiness.businessDescription}
                  </div>
                  <div className="popup-item">
                    {popupInfoBusiness.businessMobileNumber}
                  </div>
                  </div>
                </div>
              </Popup>
            )}

            {/* POPUP NOVE LOKACIJE
              ako trenutno postoji korisnik i taj korisnik ima ulogu USER */}
            {JSON.parse(sessionStorage.getItem("role") as string) ===
            UserRole.ROLE_USER ? (
              <>
              {/* //DODAJ NOVU LOKACIJU */}
                {showNewLocation ? (
                  <Popup
                    anchor="bottom"
                    longitude={Number(coordLng)}
                    latitude={Number(coordLat)}
                    closeOnClick={true}
                    onClose={() => {
                      setShowNewLocation(false);
                      formik.resetForm();
                    }}
                  >
                    <div className="addLocationContainer">
                    {errorAddLocation.length?(<div style={{color:"red"}}>{errorAddLocation}</div>):( <div className="addLocationTitle">
                        Dodajte recenziju za novu lokaciju
                      </div>)}
                     
                      <form
                        className="addLocationFormContainer"
                        onSubmit={formik.handleSubmit}
                      >
                        <div className='addNameFormContainer'>
                        <div style={{paddingTop:"8px", paddingRight:'4px'}}>Ime lokacije:</div>
                        <TextField
                          id="name"
                          name="name"
                          variant="standard"
                          sx={{ height: "20px", marginBottom: "1rem", fontSize:'small', width:'100px'}}
                          value={formik.values.name}
                          onChange={formik.handleChange}
                          error={
                            formik.touched.name && Boolean(formik.errors.name)
                          }
                          helperText={formik.touched.name && formik.errors.name}
                          autoComplete={"off"}
                        />
                        </div>
                        <div className='addNameFormContainer'>
                        <div style={{paddingTop:"8px", paddingRight:'4px'}}>Vrsta lokacije:</div>
                        <Select
                          id="selektorLocForm"
                          name="type"
                          value={formik.values.type}
                          sx={{fontSize:'small', height:'35px', width:'110px', padding:'5px'}}
                          onChange={formik.handleChange}
                          error={
                            formik.touched.type && Boolean(formik.errors.type)
                          }
                        >
                          <MenuItem style={{fontSize:'smaller'}} value="BEACH">
                            Plaža
                          </MenuItem>
                          <MenuItem style={{fontSize:'smaller'}} value="PARK">
                            Park
                          </MenuItem>
                          <MenuItem style={{fontSize:'smaller'}} value="RESTAURANT">
                            Restoran
                          </MenuItem>
                          <MenuItem style={{fontSize:'smaller'}} value="COFFEE_BAR">
                            Kafić
                          </MenuItem>
                          <MenuItem style={{fontSize:'smaller'}} value="OTHER">
                            Ostalo
                          </MenuItem>
                        </Select>
                        </div>
                        <FormControl className="addLocFormControl">
                          <div> Vaše mišljenje o lokaciji:</div>
                          <RadioGroup
                            name="rating"
                            className="radioGroupAddLocation"
                            value={formik.values.rating}
                            onChange={formik.handleChange}
                          >
                            <Radio
                              value="POSITIVE"
                              className="radioAddLocation"
                              size='small'
                            />
                            <div>Pozitivno</div>
                            <Radio
                              value="NEGATIVE"
                              className="radioAddLocation"
                              size='small'
                            />
                            <div>Negativno</div>
                          </RadioGroup>
                        </FormControl>
                          <Button
                            color="success"
                            sx={{fontSize:'small', fontFamily:'Quicksand', height:'35px'}}
                            variant="contained"
                            type="submit"
                          >
                            Dodaj lokaciju!
                          </Button>
                      </form>
                    </div>
                  </Popup>
                ) : (
                  <></>
                )}
              </>
            ) : (
              <></>
            )}
          </ReactMapGL>
        </div>

        {/* 2. BIG DIV -> SEARCH */}
        {/* <div className='search_filter_recommended'> */}
        <div className="SearchContainer">
          {/* Forma koja sadrzi search bar i filtere */}
          <form action="" method="" className="SearchForm"
            onSubmit = {(event) =>
          {
            event.preventDefault();

            addrToCoord(searchBarRef.current.value).then(coords =>
            {
              if(coords.length > 0 && mapRef) 
              {
                setCenteredLocationMarker(() =>
                {
                  return (<Marker
                    longitude= {coords[0]}
                    latitude= {coords[1]}
                    anchor="bottom"
                    z-index = {-1}
                    onClick={e => {
                      mapRef.flyTo({
                        center: [coords[0], coords[1]],
                        zoom: 15,
                        essential: true,
                        duration: 1500
                      });
                      e.originalEvent.stopPropagation();
                    }}
                  />);
                });
                mapRef.flyTo
                ({
                    center: [coords[0], coords[1]],
                    essential: true,
                    duration: 1500
                });
              }
            })
          }}>
            <input
              type="text"
              className="SearchBar"
              name="text"
              ref={searchBarRef}
              placeholder="Pretraži..."
              onChange={getSearchResults}
              autoComplete={"off"}
              required
            ></input>
            <div className="SearchResultsContainer">
                <div className='SearchResults'>
                  {searchResults}
                  {searchBarRef && searchBarRef.current && searchBarRef.current.value.trim() != "" &&
                  (
                    <div className="SearchResult" onClick= 
                    { 
                      e =>
                      {
                        addrToCoord(searchBarRef.current.value).then(coords =>
                        {
                          if(coords.length > 0 && mapRef) 
                          {
                            setCenteredLocationMarker(() =>
                            {
                              return (<Marker
                                longitude= {coords[0]}
                                latitude= {coords[1]}
                                anchor="bottom"
                                onClick={e => {
                                  if(mapRef)
                                    mapRef.flyTo({
                                      center: [coords[0], coords[1]],
                                      zoom: 15,
                                      essential: true,
                                      duration: 1500
                                    });
                                  e.originalEvent.stopPropagation();
                                }}
                              />);
                            });
                            if(mapRef)
                              mapRef.flyTo
                              ({
                                  center: [coords[0], coords[1]],
                                  essential: true,
                                  duration: 1500,
                                  zoom: 15
                              });
                          }
                        })
                      }
                    }>
                    <b>{"Pronađi"}</b>: {" '" + String(searchBarRef.current.value.trim()) + "'"}
                    </div>
                  )}
                </div>

                {/* <div className="SearchResult">Example</div> */}
              </div>

            <div className="FiltersContainer">
              <h1>Filteri</h1>

              <div className="Filters">
                <h2>Lokacije</h2>
                {locationFilters}
              </div>

                <div className="Filters">
                  <h2>Obrti</h2>
                  {businessFilters}
                </div>
              </div>
            </form>

            {/* 3. BIG DIV -> RECOMMENED */}
            <div className="Recommended">
              <h2
              style={{paddingBottom:"5px"}}>Preporučeni obrti:</h2>
              <div className="recommended-container">
                {shownBusiness?.length===0?(<div style={{paddingTop:"10px"}}>Trenutno nemamo preporučenih obrta.</div>):(<>
                {shownBusiness?.map((business) => (
                  <div
                    key={business.businessName}
                    className="recommended-item-container"
                    onClick=
                    {
                      () =>
                      {
                        if(mapRef)
                        {
                          mapRef.flyTo({
                            center: [business.longitude, business.latitude],
                            essential: true,
                            duration: 1500,
                            zoom: 15
                          });
                        }
                      }
                    }
                  >
                    <div className="recommended-item"><b>{business.businessName}</b></div>
                    <hr style={{ width: "100%", border:"1px solid var(--blueSteel)" }} />
                    <div className="recommended-item">
                      {(() => {
                        if (business.businessType === "VET") {return "Veterinarska ordinacija";
                        } else if (business.businessType === "SHOP") {return "Trgovina za pse";
                        } else if (business.businessType === "SALON") {return "Salon za uređivanje psa";
                        } else if (business.businessType === "DAYCARE") {return "Vrtić za pse";
                        } else {return "Ostalo";}
                      })()}
                    </div>
                    <div className="recommended-item">{business.businessAddress}</div>
                    <div className="recommended-item">Kontakt: {business.businessMobileNumber}</div>
                    <div className="recommended-item"><p>{business.businessDescription}</p></div>
                  </div>
                ))}</>)}
              </div>
            </div>

          </div>
        </div>
      // </div>
    );
}

export default Map;
