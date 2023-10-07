import { useState, useEffect } from "react";
import * as Yup from "yup";
import { useFormik } from "formik";
import { Link, useNavigate, useLocation} from "react-router-dom";
import TextField from "@mui/material/TextField";
import { Button } from "@mui/material";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import "./Login.css";
import { useAuth } from "../../auth/authContext";

const LoginSchema = Yup.object().shape({
  username: Yup.string().required("Obavezno"),
  password: Yup.string().required("Obavezno"),
});

const theme = createTheme({
  palette: {
    primary: {
      main: "#8fb4d6",
    },
  },
});
const Login = () => {
  const auth = useAuth();
  const navigate = useNavigate();
  const [isError, setError] = useState("");
  const [isErrorVisible, setIsErrorVisible] = useState(false);
  const [message, setMessage] = useState(false);

  const formik = useFormik({
    initialValues: {
      username: "",
      password: "",
    },
    validationSchema: LoginSchema,
    onSubmit: (values, { setErrors }) => {
      auth
        .logIn(values.username, values.password)
        .then((res) => {
          navigate("/");
        })
        .catch((err) => {
          if ((err.response && err.response.status === 401)
            || (err.response && err.response.status === 400)) {
                setIsErrorVisible(true);
                setError("Neispravno korisničko ime ili lozinka!")
          } else {
            setIsErrorVisible(true);
            setError(err.response.data.message);
            console.error(err);
          }
        });
    },
  });
  
  // popup message after registration
  const location = useLocation();

  const removeWindow = () => {
    location.state.message = null
    setMessage(false)
    navigate("/auth/login", {state: {message: null}, replace: true});
  }

  useEffect(()=> {
    if (location?.state?.message != null) {
      setMessage(true)
    }  else {
      setMessage(false)
    }
  }, [message, location?.state?.message]);

  return (
    <>
      <div className="login-main-container">
        {message? (
          <div className="popup-container-login">
            <div className="popup-login">
                <div className="registration-message-container">
                    <div className="cancel-button">
                        <div className="cancel-button-container">
                            <img className='button-icon-message' onClick={() => removeWindow()}  src={require('./../../assets/icons/cancel2.png')} alt='cancel'/>
                        </div>
                    </div>
                  <div className="registration-message">{location.state.message}
                  </div>
                </div>
            </div>
          </div>
        ):(<></>)}
        <div className="login-container">
          <div className="login-title">Dobrodošli natrag!</div>
          <form className="login-form" onSubmit={formik.handleSubmit}>
            {isErrorVisible ? (
              <div className="error-container">{isError}</div>
            ) : (
              <></>
            )}
            <TextField
              id="username"
              name="username"
              label="Korisničko ime"
              variant="outlined"
              sx={{ marginBottom: "1rem" }}
              value={formik.values.username}
              onChange={formik.handleChange}
              error={formik.touched.username && Boolean(formik.errors.username)}
              helperText={formik.touched.username && formik.errors.username}
            />
            <TextField
              id="password"
              name="password"
              type="password"
              label="Lozinka"
              variant="outlined"
              sx={{ marginBottom: "1rem" }}
              value={formik.values.password}
              onChange={formik.handleChange}
              error={formik.touched.password && Boolean(formik.errors.password)}
              helperText={formik.touched.password && formik.errors.password}
            />
            <ThemeProvider theme={theme}>
              <Button color="primary" variant="contained" type="submit">
                Prijava
              </Button>
            </ThemeProvider>
            <div className="go-to-register">
              Nemate korisnički račun?
              <Link className="go-to-link" to="/auth/register">
                Registrirajte se!
              </Link>
            </div>
          </form>
        </div>
      </div>
    </>
  );
};

export default Login;
