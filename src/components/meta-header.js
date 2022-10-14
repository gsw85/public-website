import Head from "next/head";
import ProfilePic from "../img/profilepic.jpg";

export default function MetaHeader({
  title = "Goh SW. | A Visionary and a Passionate Entrepreneur",
  description = "Dr. Goh's investment portfolio has spread across real estate developments, technology companies and civil engineering projects globally ",
  image = ProfilePic,
}) {
  return (
    <>
      <Head>
        <meta charSet="utf-8" />
        <title>{title}</title>
        <meta name="description" content={"gsw85.com | " + description} />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:site" content="@gsw85" />
        <meta name="twitter:title" content={title} />
        <meta
          name="twitter:description"
          content={"gsw85.com | " + description}
        />
        <meta name="twitter:image" content={image} />
        <meta property="og:title" content={title} />
        <meta
          property="og:description"
          content={"gsw85.com | " + description}
        />
        <meta property="og:image" content={image} />
      </Head>
    </>
  );
}
