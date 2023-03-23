import Head from "next/head";

export default function MetaHeader({
  title = "A Visionary and a Passionate Entrepreneur.",
  description = "Dr. Goh's investment portfolio boasts an impressive scope, ranging from ambitious infrastructure developments to cutting-edge technology ventures.",
  keywords = "",
  image = "/img/profilepic.jpg",
  noIndexNoFollow = true,
  url = "https://www.gsw85.com",
}) {
  const tagTitle = "Goh S.W. | " + title;
  const imageUrl = "https://www.gsw85.com" + image;
  const tagKeyWords = "gsw, goh shu wei, shu wei goh" + keywords;
  return (
    <>
      <Head>
        {noIndexNoFollow && <meta name="robots" content="noindex,nofollow" />}
        <meta charSet="utf-8" />
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="keywords" content={tagKeyWords} />
        <meta name="twitter:card" content="summary" />
        <meta property="twitter:url" content={url} />
        <meta property="twitter:title" content={tagTitle} />
        <meta property="twitter:description" content={description} />
        <meta property="twitter:image" content={imageUrl} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content={url} />
        <meta property="og:title" content={tagTitle} />
        <meta property="og:site_name" content="nosell.xyz" />
        <meta property="og:description" content={description} />
        <meta property="og:image" content={imageUrl} />
        <link rel="icon" href="/favicon.ico" type="image/x-icon" />
      </Head>
    </>
  );
}
