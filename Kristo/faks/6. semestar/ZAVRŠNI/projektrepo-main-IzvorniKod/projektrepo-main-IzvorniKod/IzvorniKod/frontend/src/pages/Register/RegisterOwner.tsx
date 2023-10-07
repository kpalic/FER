import * as Yup from "yup";
import { useState, useEffect } from "react";
import { useFormik } from "formik";
import { ThemeProvider } from "@mui/material/styles";
import { createTheme } from "@mui/material/styles";
import { MenuItem, TextField, InputLabel, FormControl } from "@mui/material";
import Select from "@mui/material/Select";
import { Button } from "@mui/material";
import { Link } from "react-router-dom";

import "./RegisterOwner.css";

import { useAuth } from "../../auth/authContext";
import { useNavigate } from "react-router-dom";
// import { getRoles } from '@testing-library/react'

const RegisterSchema = Yup.object().shape({
  username: Yup.string().required("Obavezno"),
  email: Yup.string().required("Obavezno").email("Nije važeća email adresa."),
  password: Yup.string().min(8).required("Obavezno"),
  businessName: Yup.string().required("Obavezno").max(50, "Ime obrta može imati najviše 50 znakova."),
  businessType: Yup.string().required("Obavezno"),
  businessAdress: Yup.string().required("Obavezno"),
  businessCity: Yup.string().required("Obavezno"),
  businessOIB: Yup.string()
    .required("Obavezno")
    .matches(/^[0-9]+$/, "OIB obrta mora sadržavati samo znamenke.")
    .min(11, "OIB mora imati 11 znamenki.")
    .max(11, "OIB mora imati 11 znamenki."),
  businessMobileNumber: Yup.string()
  .required("Obavezno")
  .matches(/^[0-9+-/ ]+$/, "Kontakt broja obrta ne smije sadržavati slova."),
  businessDescription: Yup.string().max(200, "Opis obrta može imati najviše 200 znakova."),
  cardNumber: Yup.string()
    .required("Obavezno")
    .matches(/^[0-9 ]+$/, "Broj kartice mora sadržavati samo znamenke.")
    .min(19, "Kartica mora imati 16 znamenki."),
  expiryDateMonth: Yup.string().required("Obavezno"),
  expiryDateYear: Yup.string().required("Obavezno"),
  cvv: Yup.string()
    .required("Obavezno")
    .matches(/^[0-9]+$/, "CVV mora sadržavati samo znamenke.")
    .min(3, "CVV mora imati minimalno 3 znamenke.")
    .max(4, "CVV mora imati maksimalno 4 znamenke."),
});

const theme = createTheme({
  palette: {
    primary: {
      main: "#8fb4d6",
    },
  },
});

const RegisterOwner = () => {
  const auth = useAuth();
  const navigate = useNavigate();

  const [isError, setError] = useState("");
  const [isErrorVisible, setIsErrorVisible] = useState(false);
  const [loading, setLoading] = useState(false)

  const formik = useFormik({
    initialValues: {
      username: "",
      email: "",
      password: "",
      businessName: "",
      businessType: "",
      businessAdress: "",
      businessCity: "",
      businessOIB: "",
      businessMobileNumber: "",
      businessDescription: "",
      cardNumber: "",
      expiryDateMonth: "",
      expiryDateYear: "",
      cvv: "",
    },
    validationSchema: RegisterSchema,
    onSubmit: (values) => {
      setLoading(true)
      auth
        .registerOwner(
          values.username,
          values.email,
          values.password,
          values.businessName,
          values.businessType,
          values.businessAdress,
          values.businessCity,
          values.businessOIB,
          values.businessMobileNumber,
          values.businessDescription,
          values.cardNumber.split(" ").join(""),
          values.expiryDateMonth,
          values.expiryDateYear,
          values.cvv
        )
        .then((res) => {
          setLoading(false)
          navigate("/auth/login", {state: {message: "Potvrdite svoju email adresu kako biste uspješno izvršili registraciju!"}, replace: true});
        })
        .catch((err) => {
          setLoading(false)
          setIsErrorVisible(true);
          setError(err.response.data.message);
          console.error(err);
        });
    },
  });


  useEffect(()=> {
    if (formik.values.cardNumber.length === 5 && formik.values.cardNumber[4] !== ' ') {
      formik.values.cardNumber = formik.values.cardNumber.substring(0, 4) + " " + formik.values.cardNumber[4];
    } else if (formik.values.cardNumber.length === 10 && formik.values.cardNumber[9] !== ' ') {
      formik.values.cardNumber = formik.values.cardNumber.substring(0, 9) + " " + formik.values.cardNumber[9];
    } else if (formik.values.cardNumber.length === 15 && formik.values.cardNumber[14] !== ' ') {
      formik.values.cardNumber = formik.values.cardNumber.substring(0, 14) + " " + formik.values.cardNumber[14];
    } else if (formik.values.cardNumber.length === 5 && formik.values.cardNumber[4] === ' ') {
      formik.values.cardNumber = formik.values.cardNumber.substring(0, 4);
    } else if (formik.values.cardNumber.length === 11 && formik.values.cardNumber[10] === ' ') {
      formik.values.cardNumber = formik.values.cardNumber.substring(0, 9);
    } else if (formik.values.cardNumber.length === 15 && formik.values.cardNumber[14] === ' ') {
        formik.values.cardNumber = formik.values.cardNumber.substring(0, 14);
    }
  }, [formik.values.cardNumber]);

  return (
    <>
      <div className="register-main-container-owner">
        <div className="register-container">
          <div className="register-title">Dobrodošli!</div>
          <form className="register-form" onSubmit={formik.handleSubmit}>
            {isErrorVisible ? (
              <div className="error-container">{isError}</div>
            ) : (
              <></>
            )}
            <div>
              <TextField
                id="username"
                name="username"
                label="Korisničko ime"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.username}
                onChange={formik.handleChange}
                error={
                  formik.touched.username && Boolean(formik.errors.username)
                }
                helperText={formik.touched.username && formik.errors.username}
              />
              <TextField
                id="email"
                name="email"
                label="Email"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.email}
                onChange={formik.handleChange}
                error={formik.touched.email && Boolean(formik.errors.email)}
                helperText={formik.touched.email && formik.errors.email}
              />
              <TextField
                id="password"
                name="password"
                label="Lozinka"
                type="password"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.password}
                onChange={formik.handleChange}
                error={
                  formik.touched.password && Boolean(formik.errors.password)
                }
                helperText={formik.touched.password && formik.errors.password}
              />

              <TextField
                id="businessName"
                name="businessName"
                label="Ime obrta"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessName}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessName &&
                  Boolean(formik.errors.businessName)
                }
                helperText={
                  formik.touched.businessName && formik.errors.businessName
                }
              />
              <FormControl fullWidth>
              <InputLabel id="businessType">Vrsta obrta</InputLabel>
              <Select
                id="businessType"
                name="businessType"
                label="Vrsta objekta"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessType}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessType &&
                  Boolean(formik.errors.businessType)
                }
              >
                <MenuItem value={0} disabled>
                  Odaberite vrstu obrta
                </MenuItem>
                <MenuItem value={"SHOP"}> Trgovina za pse </MenuItem>
                <MenuItem value={"VET"}> Veterinska ordinacija </MenuItem>
                <MenuItem value={"DAYCARE"}> Vrtić za pse </MenuItem>
                <MenuItem value={"SALON"}> Salon za uređivanje psa </MenuItem>
                <MenuItem value={"OTHER"}> Ostalo </MenuItem>
              </Select>
              </FormControl>

              <TextField
                id="businessAdress"
                name="businessAdress"
                label="Adresa obrta"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessAdress}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessAdress &&
                  Boolean(formik.errors.businessAdress)
                }
                helperText={
                  formik.touched.businessAdress && formik.errors.businessAdress
                }
              />
              <TextField
                id="businessCity"
                name="businessCity"
                label="Grad u kojem se obrt nalazi"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessCity}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessCity &&
                  Boolean(formik.errors.businessCity)
                }
                helperText={
                  formik.touched.businessCity && formik.errors.businessCity
                }
              />
              <TextField
                id="businessOIB"
                name="businessOIB"
                label="OIB obrta"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessOIB}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessOIB &&
                  Boolean(formik.errors.businessOIB)
                }
                helperText={
                  formik.touched.businessOIB && formik.errors.businessOIB
                }
              />
              <TextField
                id="businessMobileNumber"
                name="businessMobileNumber"
                label="Kontakt broj obrta"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessMobileNumber}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessMobileNumber &&
                  Boolean(formik.errors.businessMobileNumber)
                }
                helperText={
                  formik.touched.businessMobileNumber &&
                  formik.errors.businessMobileNumber
                }
              />
              <TextField
                id="businessDescription"
                name="businessDescription"
                label="Opis obrta"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.businessDescription}
                onChange={formik.handleChange}
                error={
                  formik.touched.businessDescription &&
                  Boolean(formik.errors.businessDescription)
                }
                helperText={
                  formik.touched.businessDescription &&
                  formik.errors.businessDescription
                }
              />
              <TextField
                id="cardNumber"
                name="cardNumber"
                label="Broj kartice"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.cardNumber}
                onChange={formik.handleChange}
                inputProps={{ maxLength: 19 }}
                error={
                  formik.touched.cardNumber && Boolean(formik.errors.cardNumber)
                }
                helperText={
                  formik.touched.cardNumber && formik.errors.cardNumber
                }
              />

              <div className="selektor-container">
                <div className="expiryDate-label">Datum isteka kartice: </div>
                <Select
                  className="selektor"
                  id="expiryDateMonth"
                  name="expiryDateMonth"
                  label="Mjesec"
                  value={formik.values.expiryDateMonth}
                  onChange={formik.handleChange}
                  error={
                    formik.touched.expiryDateMonth &&
                    Boolean(formik.errors.expiryDateMonth)
                  }
                >
                  <MenuItem value={0} disabled>
                    Odaberite mjesec
                  </MenuItem>
                  <MenuItem value={1}>1</MenuItem>
                  <MenuItem value={2}>2</MenuItem>
                  <MenuItem value={3}>3</MenuItem>
                  <MenuItem value={4}>4</MenuItem>
                  <MenuItem value={5}>5</MenuItem>
                  <MenuItem value={6}>6</MenuItem>
                  <MenuItem value={7}>7</MenuItem>
                  <MenuItem value={8}>8</MenuItem>
                  <MenuItem value={9}>9</MenuItem>
                  <MenuItem value={10}>10</MenuItem>
                  <MenuItem value={11}>11</MenuItem>
                  <MenuItem value={12}>12</MenuItem>
                </Select>
                <div className="slash">/</div>
                <Select
                  className="selektor"
                  labelId="getExpiryDateProp"
                  name="expiryDateYear"
                  id="getExpiryDateProps"
                  value={formik.values.expiryDateYear}
                  onChange={formik.handleChange}
                  error={
                    formik.touched.expiryDateYear &&
                    Boolean(formik.errors.expiryDateYear)
                  }
                >
                  <MenuItem value={0} id="1990" disabled>
                    Odaberite godinu
                  </MenuItem>
                  <MenuItem value={2023} id="2023">
                    2023
                  </MenuItem>
                  <MenuItem value={2024} id="2024">
                    2024
                  </MenuItem>
                  <MenuItem value={2025} id="2025">
                    2025
                  </MenuItem>
                  <MenuItem value={2026} id="2026">
                    2026
                  </MenuItem>
                  <MenuItem value={2027} id="2027">
                    2027
                  </MenuItem>
                  <MenuItem value={2028} id="2028">
                    2028
                  </MenuItem>
                  <MenuItem value={2029} id="2029">
                    2029
                  </MenuItem>
                  <MenuItem value={2030} id="2030">
                    2030
                  </MenuItem>
                  <MenuItem value={2031} id="2031">
                    2031
                  </MenuItem>
                  <MenuItem value={2032} id="2032">
                    2032
                  </MenuItem>
                  <MenuItem value={2033} id="2033">
                    2033
                  </MenuItem>
                </Select>
              </div>

              <TextField
                id="cvv"
                name="cvv"
                label="Sigurnosni kod"
                variant="outlined"
                sx={{ marginBottom: "1rem" }}
                fullWidth={true}
                value={formik.values.cvv}
                onChange={formik.handleChange}
                error={formik.touched.cvv && Boolean(formik.errors.cvv)}
                helperText={formik.touched.cvv && formik.errors.cvv}
              />

              {loading ? (<div className='loader-container'><div className='loading'><img src={require('./../../assets/icons/loading.gif')} alt='loading'/></div></div>):(<></>)}

              <ThemeProvider theme={theme}>
                <Button color="primary" variant="contained" type="submit">
                  Registracija
                </Button>
              </ThemeProvider>

              <div className="go-to-register">
                Već imate korisnički račun?
                <Link className="go-to-link" to="/auth/login">
                  Prijavite se!
                </Link>
              </div>
            </div>
          </form>
        </div>
      </div>
    </>
  );
};
export default RegisterOwner;
