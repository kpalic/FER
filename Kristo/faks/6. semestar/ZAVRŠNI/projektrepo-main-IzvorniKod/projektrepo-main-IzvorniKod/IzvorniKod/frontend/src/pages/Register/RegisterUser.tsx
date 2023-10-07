import { useState } from "react";
import * as Yup from "yup";
import { useFormik } from "formik";
import { ThemeProvider } from "@mui/material/styles";
import { createTheme } from "@mui/material/styles";
import { TextField } from "@mui/material";
import { Button } from "@mui/material";
import { Link } from "react-router-dom";

import { useAuth } from "../../auth/authContext";
import { useNavigate } from "react-router-dom";

import "./RegisterUser.css";

const RegisterSchema = Yup.object().shape({
  username: Yup.string().required("Obavezno"),
  email: Yup.string().required("Obavezno").email("Nije važeća email adresa."),
  password: Yup.string().min(8).required("Obavezno"),
});

const theme = createTheme({
  palette: {
    primary: {
      main: "#8fb4d6",
    },
  },
});

const RegisterUser = () => {
  const [isError, setError] = useState("");
  const [isErrorVisible, setIsErrorVisible] = useState(false);
  const [loading, setLoading] = useState(false)

  const auth = useAuth();
  const navigate = useNavigate();

  const formik = useFormik({
    initialValues: {
      username: "",
      email: "",
      password: "",
    },
    validationSchema: RegisterSchema,
    onSubmit: (values) => {
      setLoading(true)
      auth
        .registerUser(values.username, values.email, values.password)
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

  return (
    <>
      <div className="register-main-container">
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

              {loading ? (<div className='loader-container'><div className='loading'><img src={require('./../../assets/icons/loading.gif')} alt='loading'/></div></div>):(<></>)}
              
              <ThemeProvider theme={theme}>
                <Button color="primary" variant="contained" type="submit">
                  Registracija
                </Button>
              </ThemeProvider>
              <div className="go-to-login">
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

export default RegisterUser;
